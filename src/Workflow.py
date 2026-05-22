from altair import value
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.workflow.WorkflowManager import WorkflowManager
from scipy.stats import ttest_ind
from pyopenms import IdXMLFile
# for result section:
from pathlib import Path
from src.common.common import show_fig
from openms_insight import Table, Heatmap, LinePlot, SequenceView
from src.common.results_helpers import parse_idxml, build_spectra_cache


class Workflow(WorkflowManager):
    # Setup pages for upload, parameter, execution and results.
    # For layout use any streamlit components such as tabs (as shown in example), columns, or even expanders.
    def __init__(self) -> None:
        # Initialize the parent class with the workflow name.
        super().__init__("TOPP Workflow", st.session_state["workspace"])

    def upload(self) -> None:
        t = st.tabs(["MS data", "FASTA database"])
        with t[0]:
            # Use the upload method from StreamlitUI to handle mzML file uploads.
            self.ui.upload_widget(
                key="mzML-files",
                name="MS data",
                file_types="mzML",
                fallback=[str(f) for f in Path("example-data", "mzML").glob("*.mzML")],
            )

        with t[1]:
            self.ui.upload_widget(
                key="fasta-file",
                name="Protein FASTA database",
                file_types=("fasta", "fa"),
                fallback=[str(f) for f in Path("example-data", "db").glob("*.fasta")],
            )

    @st.fragment
    def configure(self) -> None:
        # Allow users to select mzML files for the analysis.
        self.ui.select_input_file("mzML-files", multiple=True)
        self.ui.select_input_file("fasta-file", multiple=False)

        # Create tabs for different analysis steps.
        t = st.tabs(
            ["**IsobaricAnalyzer**", "**CometAdapter**", "**PercolatorAdapter**", "**IDFilter**", "**IDMapper**", "**FileMerger**",
             "**ProteinInference**", "**IDFilter**", "**IDConflictResolver**", "**ProteinQuantifier**", "**Group Selection**"]
        )
        with t[0]:
            # Checkbox for decoy generation
            # reactive=True ensures the parent configure() fragment re-runs when checkbox changes,
            # so conditional UI (DecoyDatabase settings) updates immediately
            self.ui.input_widget(
                key="generate-decoys",
                default=True,
                name="Generate Decoy Database",
                widget_type="checkbox",
                help="Generate reversed decoy sequences for FDR calculation. Disable if your FASTA already contains decoys.",
                reactive=True,
            )

            # Reload params to get current checkbox value after it was saved
            self.params = self.parameter_manager.get_parameters_from_json()

            # Show DecoyDatabase settings if generating decoys
            if self.params.get("generate-decoys", True):
                st.info("""
                **Decoy Database Settings:**
                * **method**: How decoy sequences are generated from target protein sequences.
                  *Reverse* creates decoys by reversing each sequence, while *shuffle* randomly
                  rearranges the amino acids. Both methods preserve the amino acid composition
                  of the original protein, ensuring decoys have similar properties to real sequences
                  for accurate false discovery rate (FDR) estimation.
                """)
                self.ui.input_TOPP(
                    "DecoyDatabase",
                    custom_defaults={
                        "decoy_string": "rev_",
                        "decoy_string_position": "prefix",
                        "method": "reverse",
                    },
                    include_parameters=["method"],
                )

            comet_info = """
            **Identification (Comet):**
            * **enzyme**: The enzyme used for peptide digestion.
            * **missed_cleavages**: Number of possible cleavage sites missed by the enzyme. It has no effect if enzyme is unspecific cleavage.
            * **fixed_modifications**: Fixed modifications, specified using Unimod (www.unimod.org) terms, e.g. 'Carbamidomethyl (C)' or 'Oxidation (M)'
            * **variable_modifications**: Variable modifications, specified using Unimod (www.unimod.org) terms, e.g. 'Carbamidomethyl (C)' or 'Oxidation (M)'
            * **instrument**: Type of instrument (high_res or low_res). Use 'high_res' for high-resolution MS2 (Orbitrap, TOF), 'low_res' for ion trap.
            * **fragment_mass_tolerance**: Fragment mass tolerance for MS2 matching.
            * **fragment_bin_offset**: Offset for binning MS2 spectra. Typically 0.0 for high-res, 0.4 for low-res instruments.
            """
            if not self.params.get("generate-decoys", True):
                comet_info += """* **PeptideIndexing:decoy_string**: String that was appended (or prefixed - see 'decoy_string_position' flag below) to the accessions
                    in the protein database to indicate decoy proteins.
            """
            st.info(comet_info)

            st.write(Path(self.workflow_dir, "results"))

            comet_include = [":enzyme", "missed_cleavages", "fixed_modifications", "variable_modifications",
                             "instrument", "fragment_mass_tolerance", "fragment_error_units", "fragment_bin_offset"]
            if not self.params.get("generate-decoys", True):
                # Only show decoy_string when not generating decoys
                comet_include.append("PeptideIndexing:decoy_string")
                
            self.ui.input_TOPP(
                "IsobaricAnalyzer",
                custom_defaults={
                    "tmt11plex:reference_channel": 126,
                    "type": "tmt11plex",
                    "extraction:select_activation": "auto",
                    "extraction:reporter_mass_shift": 0.002,
                    "extraction:min_reporter_intensity": 0.0,
                    "extraction:min_precursor_purity": 0.0,
                    "extraction:precursor_isotope_deviation": 10.0,
                    "quantification:isotope_correction": "false",
                }
            )
        with t[1]:
            # Parameters for FeatureFinderMetabo TOPP tool.
            # self.ui.input_TOPP(
            #     "FeatureFinderMetabo",
            #     custom_defaults={"algorithm:common:noise_threshold_int": 1000.0},
            # )
            comet_include = [":enzyme", "missed_cleavages", "fixed_modifications", "variable_modifications",
                             "instrument", "fragment_mass_tolerance", "fragment_error_units", "fragment_bin_offset", "PeptideIndexing:IL_equivalent"]
            self.ui.input_TOPP(
                "CometAdapter",
                custom_defaults={
                    "PeptideIndexing:IL_equivalent": True,
                    "clip_nterm_methionine": "true",
                    "instrument": "high_res",
                    "missed_cleavages": 2,
                    "min_peptide_length": 6,
                    "max_peptide_length": 40,
                    "enzyme": "Trypsin/P",
                    "PeptideIndexing:unmatched_action": "warn",
                    "max_variable_mods_in_peptide": 3,
                    "precursor_mass_tolerance": 4.5,
                    "isotope_error": "0/1",
                    "precursor_error_units": "ppm",
                    "num_hits": 1,
                    "num_enzyme_termini": "fully",
                    "fragment_bin_offset": 0.0,
                    "minimum_peaks": 10,
                    "precursor_charge": "2:4",
                    "fragment_mass_tolerance": 0.015,
                    "PeptideIndexing:unmatched_action": "warn",
                    "variable_modifications": "Oxidation (M)\nAcetyl (Protein N-term)\nTMT6plex (K)\nTMT6plex (N-term)",
                    "debug": 0,
                    "force": True,
                },
                include_parameters=comet_include,
                flag_parameters=["PeptideIndexing:IL_equivalent", "force"],
                exclude_parameters=["second_enzyme"],
            )
        with t[2]:
            st.info("""
            **Filtering (IDFilter):**
            * **score:type_peptide**: Score used for filtering. If empty, the main score is used.
            * **score:psm**: The score which should be reached by a peptide hit to be kept. (use 'NAN' to disable this filter)
            """)
            self.ui.input_TOPP(
                "PercolatorAdapter",
                custom_defaults={
                    "subset_max_train": 300000,
                    "decoy_pattern": "DECOY_",
                    "score_type": "pep",
                    "post_processing_tdc": True,
                    "debug": 0,
                },
                flag_parameters=["post_processing_tdc"],
            )

        with t[3]:
            # Parameters for MetaboliteAdductDecharger TOPP tool.
            # self.ui.input_TOPP("FeatureLinkerUnlabeledKD")
            self.ui.input_TOPP(
                "IDFilter",
                custom_defaults={
                    "score:type_peptide": "q-value",
                    "score:psm": 0.10,
                },
                tool_instance_name="IDFilter-strict",
            )
        with t[4]:
            st.info("""
            **Quantification (ProteomicsLFQ):**
            * **intThreshold**: Peak intensity threshold applied in seed detection.
            * **psmFDR**: FDR threshold for sub-protein level (e.g. 0.05=5%). Use -FDR_type to choose the level. Cutoff is applied at the highest level. If Bayesian inference was chosen, it is equivalent with a peptide FDR
            * **proteinFDR**: Protein FDR threshold (0.05=5%).
            """)
            self.ui.input_TOPP(
                "IDMapper",
                custom_defaults={
                    "threads": 8,
                    "debug": 0,
                }
            )
        with t[5]:
            self.ui.input_TOPP(
                "FileMerger",
                custom_defaults={
                    "in_type": "consensusXML",
                    "append_method": "append_cols",
                    "annotate_file_origin": True,
                    "threads": 8,
                },
                flag_parameters=["annotate_file_origin"]
            )
        with t[6]:
            self.ui.input_TOPP(
                "ProteinInference",
                custom_defaults={
                    "threads": 8,
                    "picked_decoy_string": "DECOY_",
                    "picked_fdr": "true",
                    "protein_fdr": "true",
                    "Algorithm:use_shared_peptides": "true",
                    "Algorithm:annotate_indistinguishable_groups": "true",
                    "Algorithm:score_type": "PEP",
                    "Algorithm:score_aggregation_method": "best",
                    "Algorithm:min_peptides_per_protein": 1,
                }
            )
        with t[7]:
            # A single checkbox widget for workflow logic.
            # self.ui.input_widget("run-python-script", False, "Run custom Python script") *
            # Generate input widgets for a custom Python tool, located at src/python-tools.
            # Parameters are specified within the file in the DEFAULTS dictionary.
            # self.ui.input_python("example") *
            self.ui.input_TOPP(
                "IDFilter",
                custom_defaults={
                    "score:type_protein": "q-value",
                    "score:proteingroup": 0.01,
                    "score:psm": 0.01,
                    "delete_unreferenced_peptide_hits": True,
                    "remove_decoys": True
                },
                flag_parameters=["delete_unreferenced_peptide_hits", "remove_decoys"],
                tool_instance_name="IDFilter-lenient",
            )
        with t[8]:
            self.ui.input_TOPP(
                "IDConflictResolver",
                custom_defaults={
                    "threads": 4,
                }
            )

        with t[9]:
            self.ui.input_TOPP(
                "ProteinQuantifier",
                custom_defaults={
                    "method": "top",
                    "top:N": 3,
                    "top:aggregate": "median",
                    "top:include_all": True,
                    "ratios": True,
                    "threads": 8,
                    "debug": 0,
                },
                flag_parameters=["top:include_all", "ratios"]
            )
        with t[10]:
            st.markdown("### 🧪 TMT Sample Group Assignment")
            
            # 1. Determine TMT type (e.g., tmt10plex, tmt16plex)
            target_key = f"{self.parameter_manager.topp_param_prefix}IsobaricAnalyzer:1:type"
            selected_tmt = st.session_state.get(target_key, "tmt12plex")

            if "tmt" in selected_tmt:
                import re
                # Extract the number to determine the plex count
                num_plex_match = re.search(r'\d+', selected_tmt)
                if num_plex_match:
                    num_plex = int(num_plex_match.group())
                    all_channels = [f"sample{i+1}" for i in range(num_plex)]
                    
                    st.info(
                        "Enter a group name for each TMT channel.\n\n"
                        "Type **'skip'** for channels you wish to skip. (e.g., control, case, skip)"
                    )

                    # 2. Create an input_widget for each channel (automatically saved to params.json)
                    cols = st.columns(2)
                    for i, ch in enumerate(all_channels):
                        with cols[i % 2]:
                            self.ui.input_widget(
                                key=f"TMT-group-{ch}",
                                default="",
                                name=f"Group for {ch}",
                                widget_type="text",
                                help="Enter group name or 'skip' to ignore this channel.",
                            )

                    # 3. Read values from params.json and construct a dictionary in tmt_group_map format
                    # (This can be used later to filter DataFrames in subsequent logic)
                    self.params = self.parameter_manager.get_parameters_from_json()
                    
                    tmt_group_map = {}
                    for i, ch in enumerate(all_channels):
                        # Retrieve stored value (default is empty string)
                        group_val = self.params.get(f"TMT-group-{ch}", "")
                        tmt_group_map[str(i)] = group_val

                    # For data inspection (remove if not needed)
                    if st.checkbox("Show current TMT mapping"):
                        st.json(tmt_group_map)
                        
                    # 4. Clean up parameters from unused/previous TMT settings
                    all_possible_channel_keys = {f"TMT-group-{ch}" for ch in all_channels}
                    orphaned_keys = [
                        k for k in self.params.keys() 
                        if k.startswith("TMT-group-") and k not in all_possible_channel_keys
                    ]
                    
                    if orphaned_keys:
                        for key in orphaned_keys:
                            del self.params[key]
                        self.parameter_manager.save_parameters()

            else:
                st.warning("Please select a TMT type in the parameters first.")
            # with t[10]:
        #     st.markdown("### 🧪 Sample Group Assignment")
        #     target_key = f"{self.parameter_manager.topp_param_prefix}IsobaricAnalyzer:1:type"
        #     selected_tmt = st.session_state.get(target_key, "tmt12plex")
            
        #     if "tmt" in selected_tmt:
        #         import re
        #         num_plex = int(re.search(r'\d+', selected_tmt).group())
        #         all_channels = [f"sample{i+1}" for i in range(num_plex)]

        #         @st.fragment
        #         def render_group_assignment():
        #             exclude_channels = st.multiselect(
        #                 "Select samples to exclude.",
        #                 options=all_channels,
        #                 key="exclude_selector",
        #                 help="Samples selected here will be excluded from the dataframe."
        #             )

        #             st.write("exclude_channels:", exclude_channels)
                    
        #             exclude_indices = [i for i, ch in enumerate(all_channels) if ch in exclude_channels]

        #             st.write("exclude_indices:", exclude_indices)
        #             st.session_state["tmt_exclude_indices"] = exclude_indices
                    
        #             keep_channels = [ch for ch in all_channels if ch not in exclude_channels]
                    
        #             st.info("Enter group names for remaining samples.")
        #             group_mapping = {}

        #             if keep_channels:
        #                 cols = st.columns(2)
        #                 for idx, ch in enumerate(keep_channels):
        #                     original_idx = all_channels.index(ch)
        #                     with cols[idx % 2]:
        #                         group_name = st.text_input(
        #                             f"Group for {ch}",
        #                             value=st.session_state.get("tmt_group_map", {}).get(original_idx, ""),
        #                             placeholder="e.g. Control or Case",
        #                             key=f"input_{ch}_{len(keep_channels)}" 
        #                         )
        #                         group_mapping[original_idx] = group_name
        #             else:
        #                 st.warning("All samples were selected for exclusion.")
                    
        #             st.session_state["tmt_group_map"] = group_mapping

        #             if st.checkbox("Saved session data check"):
        #                 st.write("Indices to exclude:", st.session_state["tmt_exclude_indices"])
        #                 st.write("Groups for remaining indices:", st.session_state["tmt_group_map"])

        #         render_group_assignment()

        #     else:
        #         st.warning("Please select a TMT type in the parameters first.")
    
    def execution(self) -> None:
        # Any parameter checks, here simply checking if mzML files are selected
        if not self.params["mzML-files"]:
            self.logger.log("ERROR: No mzML files selected.")
            return
        
        if not self.params.get("fasta-file"):
            st.error("No FASTA file selected.")
            return False

        # Get mzML files with FileManager
        in_mzML = self.file_manager.get_files(self.params["mzML-files"])
        fasta_file = self.file_manager.get_files([self.params["fasta-file"]])[0]

        if len(in_mzML) < 1:
            st.error("At least one mzML file is required.")
            return False
        
        fasta_path = Path(fasta_file)

        self.logger.log(f"📂 Loaded {len(in_mzML)} sample(s)")

        if self.params.get("generate-decoys", True):
            decoy_fasta = fasta_path.with_suffix(".decoy.fasta")
            # Get decoy_string from DecoyDatabase params
            decoy_string = self.params.get("DecoyDatabase", {}).get("decoy_string", "rev_")

            if not decoy_fasta.exists():
                self.logger.log("🧬 Generating decoy database...")
                st.info("Generating decoy FASTA database...")
                if not self.executor.run_topp(
                    "DecoyDatabase",
                    {"in": [str(fasta_path)], "out": [str(decoy_fasta)]},
                ):
                    self.logger.log("Workflow stopped due to error")
                    return False
                self.logger.log("✅ Decoy database ready")
            st.success(f"Using decoy FASTA: {decoy_fasta.name}")
            database_fasta = decoy_fasta
        else:
            # Get decoy_string from CometAdapter params
            decoy_string = self.params.get("CometAdapter", {}).get("PeptideIndexing:decoy_string", "rev_")
            self.logger.log("📄 Using existing FASTA database")
            st.info(f"Using original FASTA: {fasta_path.name}")
            database_fasta = fasta_path

        # Log any messages.
        self.logger.log(f"Number of input mzML files: {len(in_mzML)}")

        results_dir = Path(self.workflow_dir, "results")
        iso_dir = results_dir / "isobaric_consensusXML"
        comet_dir = results_dir / "comet_results"
        perc_dir = results_dir / "percolator"
        psm_filter_dir = results_dir / "psm_filter"
        map_dir = results_dir / "idmapper"
        merge_dir = results_dir / "merged"
        protein_dir = results_dir / "protein"
        msstats_dir = results_dir / "msstats"
        quant_dir = results_dir / "quant_results"

        iso_consensus = []
        comet_results = []
        percolator_results = []
        psm_filtered = []
        mapped_ids = []

        for d in [
            iso_dir, comet_dir, perc_dir, psm_filter_dir,
            map_dir, merge_dir, protein_dir, msstats_dir, quant_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

        for mz in in_mzML:
            stem = Path(mz).stem
            iso_consensus.append(str(iso_dir / f"{stem}_iso.consensusXML"))
            comet_results.append(str(comet_dir / f"{stem}_comet.idXML"))
            percolator_results.append(str(perc_dir / f"{stem}_comet_perc.idXML"))
            psm_filtered.append(str(psm_filter_dir / f"{stem}_comet_perc_filter.idXML"))
            mapped_ids.append(str(map_dir / f"{stem}_comet_perc_filter_map.consensusXML"))

        merged_id = str(merge_dir / "ID_mapper_merge.consensusXML")
        protein_id = str(protein_dir / "ID_mapper_merge_epi.consensusXML")
        protein_filter = str(protein_dir / "ID_mapper_merge_epi_filter.consensusXML")
        protein_resolved = str(protein_dir / "ID_mapper_merge_epi_filter_resconf.consensusXML")
        # msstats_input = str(msstats_dir / "msstats_input.csv")
        consensus_out = str(quant_dir / "openms_design_protein_openms.csv") 

        # --- IsobaricAnalyzer ---
        self.logger.log("🏷️ Running isobaric labeling analysis...")
        with st.spinner("IsobaricAnalyzer"):
            if not self.executor.run_topp(
                "IsobaricAnalyzer",
                {
                    "in": in_mzML,
                    "out": iso_consensus,
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ IsobaricAnalyzer complete")

        # --- CometAdapter ---
        self.logger.log("🔎 Running peptide search...")
        with st.spinner(f"CometAdapter ({stem})"):
            comet_extra_params = {"database": str(database_fasta)}
            if self.params.get("generate-decoys", True):
                # Propagate decoy_string from DecoyDatabase
                comet_extra_params["PeptideIndexing:decoy_string"] = decoy_string
            if not self.executor.run_topp(
                "CometAdapter",
                {
                    "in": in_mzML,
                    "out": comet_results,
                },
                comet_extra_params,
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ CometAdapter complete")

        # Get fragment tolerance from CometAdapter parameters for visualization
        comet_params = self.parameter_manager.get_topp_parameters("CometAdapter")
        frag_tol = comet_params.get("fragment_mass_tolerance", 0.02)
        frag_tol_is_ppm = comet_params.get("fragment_error_units", "Da") != "Da"

        # Build visualization cache for Comet results
        results_dir_path = Path(self.workflow_dir, "results")
        cache_dir = results_dir_path / "insight_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Get mzML directory
        mzml_dir = Path(in_mzML[0]).parent

        # Build spectra cache (once, shared by all stages)
        spectra_df = None
        filename_to_index = {}

        for idxml_file in comet_results:
            idxml_path = Path(idxml_file)
            cache_id_prefix = idxml_path.stem

            # Parse idXML to DataFrame
            id_df, spectra_data = parse_idxml(idxml_path)

            # Build spectra cache (only once)
            if spectra_df is None:
                filename_to_index = {Path(f).name: i for i, f in enumerate(spectra_data)}
                spectra_df, filename_to_index = build_spectra_cache(mzml_dir, filename_to_index)

            # Initialize Table component (caches itself)
            Table(
                cache_id=f"table_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                interactivity={"file": "file_index", "spectrum": "scan_id", "identification": "id_idx"},
                column_definitions=[
                    {"field": "sequence", "title": "Sequence"},
                    {"field": "charge", "title": "Z", "sorter": "number"},
                    {"field": "mz", "title": "m/z", "sorter": "number"},
                    {"field": "rt", "title": "RT", "sorter": "number"},
                    {"field": "score", "title": "Score", "sorter": "number"},
                    {"field": "protein_accession", "title": "Proteins"},
                ],
                initial_sort=[{"column": "score", "dir": "asc"}],
                index_field="id_idx",
            )

            # Initialize Heatmap component
            Heatmap(
                cache_id=f"heatmap_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                x_column="rt",
                y_column="mz",
                intensity_column="score",
                interactivity={"identification": "id_idx"},
            )

            # Initialize SequenceView component
            seq_view = SequenceView(
                cache_id=f"seqview_{cache_id_prefix}",
                sequence_data=id_df.lazy().select(["id_idx", "sequence", "charge", "file_index", "scan_id"]).rename({
                    "id_idx": "sequence_id",
                    "charge": "precursor_charge",
                }),
                peaks_data=spectra_df.lazy(),
                filters={
                    "identification": "sequence_id",
                    "file": "file_index",
                    "spectrum": "scan_id",
                },
                interactivity={"peak": "peak_id"},
                cache_path=str(cache_dir),
                deconvolved=False,
                annotation_config={
                    "ion_types": ["b", "y"],
                    "neutral_losses": True,
                    "tolerance": frag_tol,
                    "tolerance_ppm": frag_tol_is_ppm,
                },
            )

            # Initialize LinePlot from SequenceView
            LinePlot.from_sequence_view(
                seq_view,
                cache_id=f"lineplot_{cache_id_prefix}",
                cache_path=str(cache_dir),
                title="Annotated Spectrum",
                styling={
                    "unhighlightedColor": "#CCCCCC",
                    "highlightColor": "#E74C3C",
                    "selectedColor": "#F3A712",
                },
            )

        self.logger.log("✅ Peptide search complete")

        # --- PercolatorAdapter ---
        self.logger.log("📊 Running rescoring...")
        with st.spinner(f"PercolatorAdapter"):
            if not self.executor.run_topp(
                "PercolatorAdapter",
                {
                    "in": comet_results,
                    "out": percolator_results,
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        # Build visualization cache for Percolator results
        for idxml_file in percolator_results:
            idxml_path = Path(idxml_file)
            cache_id_prefix = idxml_path.stem

            # Parse idXML to DataFrame
            id_df, spectra_data = parse_idxml(idxml_path)

            # Initialize Table component (caches itself)
            Table(
                cache_id=f"table_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                interactivity={"file": "file_index", "spectrum": "scan_id", "identification": "id_idx"},
                column_definitions=[
                    {"field": "sequence", "title": "Sequence"},
                    {"field": "charge", "title": "Z", "sorter": "number"},
                    {"field": "mz", "title": "m/z", "sorter": "number"},
                    {"field": "rt", "title": "RT", "sorter": "number"},
                    {"field": "score", "title": "Score", "sorter": "number"},
                    {"field": "protein_accession", "title": "Proteins"},
                ],
                initial_sort=[{"column": "score", "dir": "asc"}],
                index_field="id_idx",
            )

            # Initialize Heatmap component
            Heatmap(
                cache_id=f"heatmap_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                x_column="rt",
                y_column="mz",
                intensity_column="score",
                interactivity={"identification": "id_idx"},
            )

            # Initialize SequenceView component
            seq_view = SequenceView(
                cache_id=f"seqview_{cache_id_prefix}",
                sequence_data=id_df.lazy().select(["id_idx", "sequence", "charge", "file_index", "scan_id"]).rename({
                    "id_idx": "sequence_id",
                    "charge": "precursor_charge",
                }),
                peaks_data=spectra_df.lazy(),
                filters={
                    "identification": "sequence_id",
                    "file": "file_index",
                    "spectrum": "scan_id",
                },
                interactivity={"peak": "peak_id"},
                cache_path=str(cache_dir),
                deconvolved=False,
                annotation_config={
                    "ion_types": ["b", "y"],
                    "neutral_losses": True,
                    "tolerance": frag_tol,
                    "tolerance_ppm": frag_tol_is_ppm,
                },
            )

            # Initialize LinePlot from SequenceView
            LinePlot.from_sequence_view(
                seq_view,
                cache_id=f"lineplot_{cache_id_prefix}",
                cache_path=str(cache_dir),
                title="Annotated Spectrum",
                styling={
                    "unhighlightedColor": "#CCCCCC",
                    "highlightColor": "#E74C3C",
                    "selectedColor": "#F3A712",
                },
            )

        self.logger.log("✅ PercolatorAdapter complete")
            
        # --- IDFilter ---
        self.logger.log("🔧 Filtering identifications...")
        with st.spinner(f"IDFilter"):
            if not self.executor.run_topp(
                "IDFilter",
                {
                    "in": percolator_results,
                    "out": psm_filtered,
                },
                tool_instance_name="IDFilter-strict"
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ IDFilter-strict complete")

        # Build visualization cache for Filter results
        for idxml_file in psm_filtered:
            idxml_path = Path(idxml_file)
            cache_id_prefix = idxml_path.stem

            # Parse idXML to DataFrame
            id_df, spectra_data = parse_idxml(idxml_path)

            # Initialize Table component (caches itself)
            Table(
                cache_id=f"table_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                interactivity={"file": "file_index", "spectrum": "scan_id", "identification": "id_idx"},
                column_definitions=[
                    {"field": "sequence", "title": "Sequence"},
                    {"field": "charge", "title": "Z", "sorter": "number"},
                    {"field": "mz", "title": "m/z", "sorter": "number"},
                    {"field": "rt", "title": "RT", "sorter": "number"},
                    {"field": "score", "title": "Score", "sorter": "number"},
                    {"field": "protein_accession", "title": "Proteins"},
                ],
                initial_sort=[{"column": "score", "dir": "asc"}],
                index_field="id_idx",
            )

            # Initialize Heatmap component
            Heatmap(
                cache_id=f"heatmap_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                x_column="rt",
                y_column="mz",
                intensity_column="score",
                interactivity={"identification": "id_idx"},
            )

            # Initialize SequenceView component
            seq_view = SequenceView(
                cache_id=f"seqview_{cache_id_prefix}",
                sequence_data=id_df.lazy().select(["id_idx", "sequence", "charge", "file_index", "scan_id"]).rename({
                    "id_idx": "sequence_id",
                    "charge": "precursor_charge",
                }),
                peaks_data=spectra_df.lazy(),
                filters={
                    "identification": "sequence_id",
                    "file": "file_index",
                    "spectrum": "scan_id",
                },
                interactivity={"peak": "peak_id"},
                cache_path=str(cache_dir),
                deconvolved=False,
                annotation_config={
                    "ion_types": ["b", "y"],
                    "neutral_losses": True,
                    "tolerance": frag_tol,
                    "tolerance_ppm": frag_tol_is_ppm,
                },
            )

            # Initialize LinePlot from SequenceView
            LinePlot.from_sequence_view(
                seq_view,
                cache_id=f"lineplot_{cache_id_prefix}",
                cache_path=str(cache_dir),
                title="Annotated Spectrum",
                styling={
                    "unhighlightedColor": "#CCCCCC",
                    "highlightColor": "#E74C3C",
                    "selectedColor": "#F3A712",
                },
            )

        # --- IDMapper ---
        self.logger.log("🗺️ Mapping IDs to isobaric consensus features...")
        for iso, psm, mapped in zip(iso_consensus, psm_filtered, mapped_ids):
            iso_stem = Path(iso).stem
            with st.spinner(f"IDMapper ({iso_stem})"):
                if not self.executor.run_topp(
                    "IDMapper",
                    {
                        "in": [iso],
                        "id": [psm],
                        "out": [mapped],
                    },
                ):
                    self.logger.log("Workflow stopped due to error")
                    return False
        self.logger.log("✅ IDMapper complete")

        # --- FileMerger ---
        self.logger.log("🔗 Merging mapped consensus files...")
        with st.spinner("FileMerger"):
            if not self.executor.run_topp(
                "FileMerger",
                {
                    "in": mapped_ids,
                    "out": [merged_id],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ FileMerger complete")

        # --- ProteinInference ---
        self.logger.log("🧩 Running protein inference...")
        with st.spinner("ProteinInference"):
            if not self.executor.run_topp(
                "ProteinInference",
                {
                    "in": [merged_id],
                    "out": [protein_id],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ ProteinInference complete")

        # --- IDFilter-lenient (Protein) ---    
        self.logger.log("🔬 Filtering proteins...")
        with st.spinner("IDFilter (Protein)"):
            if not self.executor.run_topp(
                "IDFilter",
                {
                    "in": [protein_id],
                    "out": [protein_filter],
                },
                tool_instance_name="IDFilter-lenient"
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ IDFilter-lenient (Protein) complete")

        # ================================
        # ✨ NEW: 8️⃣ IDConflictResolver (protein_filter → protein_resolved)
        # ================================
        self.logger.log("⚖️ Resolving ID conflicts...")
        with st.spinner("IDConflictResolver"):
            if not self.executor.run_topp(
                "IDConflictResolver",
                {
                    "in": [protein_filter],
                    "out": [protein_resolved],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ IDConflictResolver complete")

        # ================================
        # ✨ NEW: 🔟 ProteinQuantifier (protein_resolved → consensus_out)
        # ================================
        self.logger.log("📐 Running protein quantification...")
        with st.spinner("ProteinQuantifier"):
            if not self.executor.run_topp(
                "ProteinQuantifier",
                {
                    "in": [protein_resolved],
                    "out": [consensus_out],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ ProteinQuantifier complete")
        self.logger.log("🎉 WORKFLOW FINISHED")

    @st.fragment
    def results(self) -> None:
        st.title("📊 DDA-TMT Analysis Results")

        # Tab configuration (TMT-specific)
        tabs = st.tabs([
            "🔍 Identification", 
            "🔍 Rescoring & Filter", 
            "📊 TMT Reporter Intensity", 
            "🧬 Protein Grouping", 
            "🌋 Statistical Analysis"
        ])
        
        id_tab, filter_tab, tmt_tab, prot_tab, stat_tab = tabs

        # Helper: idXML to DataFrame (Maintain existing code)
        def idxml_to_df(idxml_file):
            proteins, peptides = [], []
            IdXMLFile().load(str(idxml_file), proteins, peptides)
            records = []
            for pep in peptides:
                rt, mz = pep.getRT(), pep.getMZ()
                for h in pep.getHits():
                    records.append({
                        "RT": rt, "m/z": mz, "Sequence": h.getSequence().toString(),
                        "Charge": h.getCharge(), "Score": h.getScore(),
                        "Proteins": ",".join([ev.getProteinAccession() for ev in h.getPeptideEvidences()])
                    })
            return pd.DataFrame(records)

        # 1. Identification (Comet)
        with id_tab:
            comet_files = sorted(Path(self.workflow_dir, "results", "comet").glob("*.idXML"))
            if comet_files:
                selected_file = st.selectbox("Select Identification file", comet_files, key="comet_sb")
                df_comet = idxml_to_df(selected_file)
                st.dataframe(df_comet, use_container_width=True)
                # Scatter plot code can remain the same

        # 2. Filtering (Percolator + IDFilter)
        with filter_tab:
            filter_files = sorted(Path(self.workflow_dir, "results", "psm_filter").glob("*.idXML"))
            if filter_files:
                selected_f = st.selectbox("Select Filtered file", filter_files, key="filter_sb")
                df_filter = idxml_to_df(selected_f)
                st.success(f"Identified {len(df_filter)} PSMs after filtering (FDR < 0.01)")
                st.dataframe(df_filter, use_container_width=True)

        # 3. TMT Reporter Intensity (IsobaricAnalyzer)
        with tmt_tab:
            st.subheader("TMT Reporter Ion Intensity Distribution")
            # Extract channel-specific intensities from IsobaricAnalyzer consensusXML
            iso_files = sorted(Path(self.workflow_dir, "results", "isobaric_consensusXML").glob("*.consensusXML"))
            if iso_files:
                sel_iso = st.selectbox("Select TMT result", iso_files)
                # For simple visualization, it is better to use the quantitative results CSV if available
                # Example distribution using the final CSV (openms_design_protein_openms.csv)
                quant_file = Path(self.workflow_dir, "results", "quant", "openms_design_protein_openms.csv")
                if quant_file.exists():
                    df_q = pd.read_csv(quant_file)
                    # Filter channel columns (usually prefixed with 'Abundance_' or specific tags)
                    # Column names need to be verified based on the OpenMS output structure
                    intensity_cols = [c for c in df_q.columns if 'intensity' in c.lower() or 'abundance' in c.lower()]
                    if intensity_cols:
                        fig_box = px.box(df_q.melt(value_vars=intensity_cols), x="variable", y="value", log_y=True,
                                        title="Log-scaled Reporter Intensity Distribution per Channel")
                        st.plotly_chart(fig_box)

        # 4. Protein Grouping & Quantification
        with prot_tab:
            st.subheader("🧬 Final Protein-Level Results")
            final_csv = Path(self.workflow_dir, "results", "quant", "openms_design_protein_openms.csv")
            
            if final_csv.exists():
                df_final = pd.read_csv(final_csv)
                st.info(f"Total Protein Groups: {df_final['ProteinName'].nunique() if 'ProteinName' in df_final.columns else len(df_final)}")
                st.dataframe(df_final, use_container_width=True)
                
                # CSV Download Button
                st.download_button("Download Results", df_final.to_csv(index=False), "TMT_Results.csv")
            else:
                st.warning("Final quantification CSV not found.")

        # 5. Statistical Analysis (Volcano Plot etc.)
        with stat_tab:
            final_csv = Path(self.workflow_dir, "results", "quant", "openms_design_protein_openms.csv")
            
            if not final_csv.exists():
                st.warning("Analysis results not found. Please run the workflow first.")
                return

            try:
                # 1️⃣ Data loading and preprocessing
                df_quant = pd.read_csv(final_csv)
                
                # Identify intensity (abundance) columns from TMT results
                # Typically starts with 'abundance_' or 'intensity_' depending on OpenMS output format
                intensity_cols = [c for c in df_quant.columns if 'abundance' in c.lower() or 'intensity' in c.lower()]
                
                if len(intensity_cols) < 2:
                    st.error("Not enough intensity columns found for comparison.")
                    return

                # 2️⃣ Group setup (use existing source's group_map)
                # In TMT, multiple channels (samples) exist within a single file,
                # so map which column belongs to which group (Control/Case).
                st.subheader("Group Comparison Setup")
                col1, col2 = st.columns(2)
                with col1:
                    group_a_cols = st.multiselect("Select Control Group Channels", intensity_cols, default=[intensity_cols[0]])
                with col2:
                    group_b_cols = st.multiselect("Select Case Group Channels", intensity_cols, default=[intensity_cols[-1]])

                if st.button("Run Statistical Analysis"):
                    stats_results = []
                    
                    for _, row in df_quant.iterrows():
                        g1 = row[group_a_cols].values.astype(float)
                        g2 = row[group_b_cols].values.astype(float)
                        
                        # Calculate Log2 Fold Change
                        log2fc = np.log2(np.mean(g2) / np.mean(g1)) if np.mean(g1) > 0 else 0
                        
                        # T-test (p-value)
                        _, pval = ttest_ind(g1, g2, nan_policy='omit')
                        
                        stats_results.append({
                            "Protein": row.get("ProteinName", "Unknown"),
                            "log2FC": log2fc,
                            "pvalue": pval,
                            "-log10_pvalue": -np.log10(pval) if pval > 0 else 0
                        })
                    
                    df_stats = pd.DataFrame(stats_results)

                    # 3️⃣ Volcano plot visualization
                    st.divider()
                    st.subheader("Volcano Plot")
                    
                    # Define colors to highlight significant proteins
                    df_stats['Significance'] = 'Normal'
                    df_stats.loc[(df_stats['log2FC'] > 1) & (df_stats['pvalue'] < 0.05), 'Significance'] = 'Up'
                    df_stats.loc[(df_stats['log2FC'] < -1) & (df_stats['pvalue'] < 0.05), 'Significance'] = 'Down'

                    fig_volcano = px.scatter(
                        df_stats, x="log2FC", y="-log10_pvalue",
                        color="Significance",
                        hover_data=["Protein"],
                        color_discrete_map={'Up': 'red', 'Down': 'blue', 'Normal': 'gray'},
                        title=f"Comparison: {', '.join(group_b_cols)} vs {', '.join(group_a_cols)}"
                    )
                    
                    # Guidelines (p=0.05, FC=2)
                    fig_volcano.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="black")
                    fig_volcano.add_vline(x=1, line_dash="dash", line_color="black")
                    fig_volcano.add_vline(x=-1, line_dash="dash", line_color="black")

                    st.plotly_chart(fig_volcano, use_container_width=True)
                    st.dataframe(df_stats.sort_values("pvalue"), use_container_width=True)

            except Exception as e:
                st.error(f"Error during statistical analysis: {e}")