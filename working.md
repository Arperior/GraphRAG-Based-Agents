Ban raha hai sabar karo

1. Entry & Configuration

    app.py (Streamlit UI) starts.

    config.load_config() loads:

        Neo4j, Gemini, local Mistral (GGUF), LLaVA, YOLO configs.

    setup_logging() configures rotating file + console logging.

    Neo4j setup:

        neo4j_client.init_indexes()

        neo4j_client.check_apoc()

    User:

        Selects/creates USER_ID.

        memory.ensure_user_exists(USER_ID).

2. Ingestion Pipelines
2.1 Text Ingestion

UI: “Ingest Data Into Graph → Text Input”

    User pastes text_input → click “Ingest Text”.

    preprocessing.chunk_tokens(text_input) → list of chunks.

    For each chunk c:

        entity_extraction.extract_graph(c):

            spaCy → spacy_candidates(c) (optional seeds).

            Prompt extract_graph.txt → llm_client_local.generate_json() (Mistral via gpu_manager).

            json_utils.parse_llm_graph / parse_llm_json_list → {entities, relations}.

        Optional extra relations:

            relation_extractor.extract_relations_from_text(c):

                Prompt extract_relations.txt → generate_json() → cleaned list of relations.

        Combine base_rel + refined_relations.

        graph_builder.build_and_store_graph():

            Normalizes entities/relations.

            Calls neo4j_client.store_chunk_with_graph():

                MERGE (c:Chunk)

                MERGE (e:Entity) + (:Entity)-[:MENTIONED_IN]->(:Chunk)

                MERGE (a:Entity)-[:RELATION {…}]->(b:Entity)

            memory.record_user_chunk_interest(USER_ID, chunk_id, query="", tokens=[]).

Result: Text chunks, entities, and relations stored in Neo4j, linked to the user via (:User)-[:INTERESTED_IN]->(:Chunk).
2.2 PDF Ingestion

UI: “Ingest Data Into Graph → PDF Upload”

    Upload PDF → click “Process PDF”.

    pdf_utils.process_pdf_upload(uploaded_file, run_relations, USER_ID):

        extract_text_from_pdf_bytes() (PyPDF2) → full_text.

        summarize_with_gemini(full_text):

            Gemini (GenerativeModel) with retry/backoff.

            If fails/short → fallback to full_text.

        Save target_text + metadata as .txt in data/txt/.

        preprocessing.chunk_tokens(target_text) → chunks.

        For each chunk:

            entity_extraction.extract_graph(chunk) (Mistral).

            Optional relation_extractor.extract_relations_from_text(chunk).

            graph_builder.build_and_store_graph(chunk_id, chunk, entities, relations_chunk, user_id=USER_ID, source=f"pdf:{filename}").

        Return summary, stats to UI.

Result: Summarized (or full) PDF content converted to chunks → entities & relations → stored in Neo4j, linked to user and PDF source.
2.3 Image / Vision Ingestion (Multimodal GraphRAG)

UI: “Ingest Data Into Graph → Image Upload”

    Upload image → optional user_context → click “Analyze & Ingest Image”.

    Streamlit saves image to _cfg.uploads_dir.

    llm_client_vision.process_image_pipeline(image_path, user_context) (from that module; high-level steps):

        segment_image(image_path):

            gpu_manager.request_permission_to_load("yolo").

            Load YOLO model (_get_yolo_model()).

            Run segmentation:

                If detections → create ImageFeatureBlocks (per object).

                Else → _create_grid_blocks() 2×2 grid fallback.

        generate_block_descriptions(blocks):

            gpu_manager.request_permission_to_load("vision").

            Load LLaVA (_get_llava_model()).

            For each block:

                Choose prompt (describe_feature_block vs describe_document_block).

                LLaVA chat completion → set block.description.

        extract_global_entities(image_path, user_context):

            LLaVA with extract_scene_graph.txt → JSON scene graph: {summary, entities, relations}.

    graph_builder.build_and_store_image(image_path, scene_graph, USER_ID, user_context):

        Deterministic img_id from filename.

        Calls neo4j_client.store_image_scene_graph():

            MERGE (i:Image {id}) with path, summary, user_context.

            MERGE (n:Entity {name}) (modality visual) + (:Image)-[:DEPICTS]->(:Entity).

            MERGE (a:Entity)-[:RELATION {relation:VISUAL_REL}]->(b:Entity) for visual relations.

            (:User)-[:INTERESTED_IN]->(:Image).

    Cross-modal Fusion:

        _perform_fusion_check(visual_entities, context=user_context or "..."):

            For each visual entity:

                search_potential_matches(name) (Neo4j) → candidate text entities.

                Prompt entity_resolution.txt → llm_client_local.generate_json() (Mistral).

                If match_found and target_name ∈ candidates:

                    merge_entities(target, name) → unify visual and text entities.

Result: Images become nodes in an N-MMKG; visual entities and relations stored and softly aligned with text entities via Mistral-based fusion.
3. Graph Enrichment & Communities
3.1 Graph Enrichment (LLM-suggested Links)

UI: “Run Graph Enrichment” (from app.py, via suggest_and_create_links(USER_ID)).

    enrichment_graph.get_degree_thresholds(USER_ID):

        From user’s reachable entities, compute degree percentiles (p25 = orphans, p75 = anchors).

    suggest_and_create_links(USER_ID, confidence_threshold, ...):

        Query relevant entities (orphan/anchor candidates), filter noise (icons, pure numbers, etc.).

        Build orphan and anchor lists for prompt.

        Prompt graph_enrichment.txt → llm_client_local.generate_json() → candidate relations.

        Filter:

            confidence >= threshold, non-empty, non-self, deduplicated by (src, tgt, relation).

        UNWIND write query:

            MATCH (a:Entity {name:source}), MATCH (b:Entity {name:target}).

            WHERE NOT (a)-[]-(b) (no existing edge).

            MERGE (a)-[rel:RELATED {type: relation}]->(b) with metadata.

Result: New high-confidence, user-specific enrichment edges between entities.
3.2 Community Detection & Summarization

    From UI, when triggered:

        clustering.run_leiden(resolution, user_id=USER_ID):

            _export_entities_and_edges(user_id):

                Nodes: Entities + Images reachable via user’s subgraph.

                Edges:

                    Semantic RELATION|RELATED (weighted by confidence).

                    Image–Entity DEPICTS edges (strong weight).

                    Text co-occurrence edges via shared Chunk (weight ∝ co-occurrence).

            Build weighted igraph graph.

            Run Leiden (leidenalg) → partition.

            Write results back to Neo4j:

                Community nodes (:Community) and (:Entity)/(:Image)-[:IN_COMMUNITY]->(:Community) for the user.

    clustering.summarize_communities(USER_ID, ...) (called in app):

        Uses gemini_complete() for each community:

            Gather entities/chunks per community.

            Generate human-readable community summaries.

Result: User-personalized topic communities with Gemini summaries for global sense-making over the KG.
4. Retrieval & Question Answering

UI: Main chat area in app.py.

    User enters query q.

    retrieval.gather_evidence_for_query(q, USER_ID, ...) (not fully printed above, but structure is implied):

        find_entities(q, top_k, USER_ID):

            Personal-scope entity search:

                From (:User {id})-[:INTERESTED_IN]->(src),

                src-[:MENTIONED_IN|DEPICTS]->(root:Entity),

                root-[:RELATION|RELATED*0..2]-(e:Entity).

                Filter by token-based Cypher where clauses.

            If no USER_ID, global MATCH (e:Entity).

            Scoring per candidate:

                _score_entity_name() (exact/prefix/substring + Levenshtein).

                _overlap_score() on descriptions.

                _boost_by_user_memory():

                    memory.get_user_recent_queries()

                    memory.get_user_longterm_memory_text()

                    memory.get_user_interest_count().

                _modality_bias() (boost visual when query mentions images).

                De-emphasize visual-only entities for non-visual queries.

        From top entities, call k_hop_chunks() (Neo4j helper) to pull text chunks and nearby nodes (k-hop evidence).

    retrieval.synthesize_answer(q, evidence, USER_ID, ..):

        Compose a prompt with:

            Query + selected entities.

            k-hop evidence chunks (possibly truncated via utils.truncate()).

            User long-term memory context (memory.get_user_longterm_memory_text()).

        Call gemini_complete(prompt) to generate a natural language answer.

    Store memory:

        memory.store_query_and_answer(USER_ID, q, answer).

        memory.record_user_chunk_interest(USER_ID, chunk_id, q, tokens) for retrieved chunks (helps future personalization).

    UI displays:

        System messages + answer.

        Optionally shows debug info (entities, evidence chunks, etc., depending on how you’ve wired the UI).

Result: Strictly personal GraphRAG retrieval with:

    Multi-hop (k-hop) evidence from Neo4j.

    Multimodal entities in the same KG.

    User memory and interests feeding back into scoring.

5. GPU / Model Coordination

Underlying all LLM/vision calls:

    gpu_manager.request_permission_to_load(model_type):

        If another model type is active:

            .close() if possible, del, gc.collect(), torch.cuda.empty_cache(), torch.cuda.ipc_collect().

    gpu_manager.register_model(model_obj, model_type):

        Keeps a single active model (type "text", "vision", or "yolo") to avoid VRAM conflicts.

Used by:

    llm_client_local (Mistral, "text").

    llm_client_vision (LLaVA "vision", YOLO "yolo").
