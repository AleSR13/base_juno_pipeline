from __future__ import annotations

"""The Juno pipeline library contains the basic classes to build a bacteriology
genomics pipeline with the format used in the IDS- bioinformatics group at the
RIVM.

All of our pipelines use Snakemake.
"""

import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from pandas import read_csv
from snakemake import snakemake

from juno_library.helper_functions import (
    message_formatter,
    error_formatter,
    SnakemakeKwargsAction,
    validate_file_has_min_lines,
    get_commit_git,
    get_repo_url,
)
from typing import Any, Optional, Dict, Tuple, cast, List, Union
import argparse


@dataclass()
class Pipeline:
    """Class to perform actions that need to be done before running a pipeline.

    This class checks that input directory exists and has the expected
    input files, generates a dictionary (sample_dict) with sample names
    and their corresponding files, makes a dictionary with metadata if
    necessary. It has been written to be adapted to different pipelines
    accepting fastq and/or fasta files as input.
    """

    pipeline_name: str
    pipeline_version: str

    input_type: Union[str, Tuple[str, ...]] = "both"
    fasta_dir: Optional[Path] = None
    fastq_dir: Optional[Path] = None
    vcf_dir: Optional[Path] = None
    exclusion_file: Optional[Path] = None
    juno_metadata: Optional[dict[str, Any]] = None

    excluded_samples: set[str] = field(default_factory=set)
    min_num_lines: int = -1

    # Setup some audit trail params
    date_and_time: str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    unique_id: UUID = uuid4()
    hostname: str = str(subprocess.check_output(["hostname"]).strip().decode("utf-8"))

    # These are passed to snakemake
    snakefile: str = "Snakefile"
    sample_sheet: Path = pathlib.Path("config/sample_sheet.yaml").resolve()
    user_parameters: dict[str, Any] = field(default_factory=dict)
    # user_parameters is created during the pipeline run to start snakemake with
    user_parameters_file: Path = pathlib.Path("config/user_parameters.yaml").resolve()
    # snakemake_config should be specified in the pipeline repository and has default values
    snakemake_config: dict[str, Any] = field(default_factory=dict)
    snakemake_args: dict[str, Any] = field(
        default_factory=lambda: dict(
            cores=300,
            nodes=300,
            force_incomplete=True,
            use_conda=False,
            conda_prefix=None,
            use_singularity=True,
            singularity_args="",
            singularity_prefix=None,
            restart_times=0,
            latency_wait=60,
            conda_frontend="mamba",
            keepgoing=True,
            printshellcmds=True,
        )
    )

    parser: argparse.ArgumentParser = field(default_factory=argparse.ArgumentParser)
    argv: list[str] = field(
        default_factory=lambda: [x for x in sys.argv if not x.endswith(".py")]
    )

    def __post_init__(
        self,
    ) -> None:
        # TODO: remove this line when self.input_type is a tuple in all pipelines
        if isinstance(self.input_type, str):
            assert self.input_type in [
                "fastq",
                "fasta",
                "both",
                "vcf",
                "bam",
                "fastq_and_fasta",
                "fastq_and_vcf",
                "bam_and_vcf",
            ], "if input_type is a str, the value can only be 'fastq', 'fasta', 'vcf', 'bam', 'both'/'fastq_and_fasta', 'fastq_and_vcf' or 'bam_and_vcf'"
        elif isinstance(self.input_type, tuple):
            assert all(
                [x in ["fastq", "fasta", "vcf", "bam"] for x in self.input_type]
            ), "if input_type is a tuple, the values can only be 'fastq', 'fasta', 'vcf' or 'bam'"

        self.snakemake_config["sample_sheet"] = str(self.sample_sheet)
        self.add_argument = self.parser.add_argument
        self._add_args_to_parser()

    def setup(self) -> None:
        """Parse arguments, create and validate sample_dict.

        Raises:
            FileNotFoundError | KeyError | ValueError: Might raise an error if the input_dir is not found or does not all files that are needed.
        """
        self._parse_args()

        self.__set_exluded_samples()
        self.snakemake_args["singularity_args"] = (
            f"--bind {self.input_dir}:{self.input_dir} --bind {self.output_dir}:{self.output_dir}"
            if self.snakemake_args["use_singularity"]
            else ""
        )

        try:
            print(
                message_formatter(
                    "Making a list of samples to be processed in this pipeline run..."
                )
            )
            self.__build_sample_dict()
        except FileNotFoundError as e:
            assert (
                self.input_dir.is_dir()
            ), f"The provided input directory ({str(self.input_dir)}) does not exist. Please provide an existing directory"
            raise e

        print(
            message_formatter(
                "Validating that all expected input files per sample are present in the input directory..."
            )
        )
        self.__validate_sample_dict()

        # Validate input files
        assert (
            self.input_dir.is_dir()
        ), f"The provided input directory ({str(self.input_dir)}) does not exist. Please provide an existing directory"

    def run(self) -> None:
        """Setup and run pipeline using snakemake.

        It has all the pre-determined input for running a Juno pipeline.
        Everything is customizable to run outside the RIVM but the
        defaults are set for the RIVM and especially for an LSF cluster.
        The commands are for now set to use bsub so it will not work
        with other types of clusters but it is on the to-do list to do
        it.
        """
        self.setup()
        self.sample_sheet.parent.mkdir(exist_ok=True, parents=True)
        with open(self.sample_sheet, "w") as f:
            yaml.dump(self.sample_dict, f)

        self.user_parameters_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.user_parameters_file, "w") as f:
            yaml.dump(self.user_parameters, f)
        print(message_formatter(f"Running {self.pipeline_name} pipeline."))

        # Generate pipeline audit trail only if not dryrun (or unlock)
        # store the exclusion file in the audit_trail as well
        if not self.dryrun or self.unlock:
            self.audit_trail_files = self._generate_audit_trail()

        if self.local:
            print(message_formatter("Jobs will run locally"))
            cluster = None
        else:
            print(message_formatter("Jobs will be sent to the cluster"))
            cluster_log_dir = pathlib.Path(str(self.output_dir)).joinpath(
                "log", "cluster"
            )
            cluster_log_dir.mkdir(parents=True, exist_ok=True)
            cluster = (
                'bsub -q {resources.queue} \
                    -n {threads} \
                    -o %s/{name}_{wildcards}_{jobid}.out \
                    -e %s/{name}_{wildcards}_{jobid}.err \
                    -R "span[hosts=1]" \
                    -R "rusage[mem={resources.mem_gb}G]" \
                    -M {resources.mem_gb}G \
                    -W %s '
                % (
                    str(cluster_log_dir),
                    str(cluster_log_dir),
                    str(self.time_limit),
                )
            )
            self.snakemake_args["cluster"] = cluster

        self.snakemake_args["jobname"] = self.pipeline_name + "_{name}.jobid{jobid}"

        pipeline_run_successful: bool = snakemake(
            self.snakefile,
            workdir=str(self.workdir),
            config=self.snakemake_config,
            configfiles=[self.user_parameters_file],
            unlock=self.unlock,
            dryrun=self.dryrun,
            **self.snakemake_args,
        )

        assert pipeline_run_successful, error_formatter(
            f"An error occured while running the snakemake part of the {self.pipeline_name} pipeline. Check the logs."
        )
        if not (self.dryrun or self.unlock):
            _snakemake_report_run_succesful = self._make_snakemake_report()
        print(message_formatter(f"Finished running {self.pipeline_name} pipeline!"))

    def _add_args_to_parser(self) -> None:
        """Add arguments to self.parser."""
        self.add_argument(
            "-i",
            "--input",
            type=Path,
            metavar="DIR",
            required=True,
            help="Relative or absolute path to the input directory. It must contain all the raw reads (fastq) files for all samples to be processed (not in subfolders).",
        )
        self.add_argument(
            "-o",
            "--output",
            type=Path,
            metavar="DIR",
            default=Path("output"),
            help="Relative or absolute path to the output directory. If none is given, an 'output' directory will be created in the current directory.",
        )
        self.add_argument(
            "-w",
            "--workdir",
            type=Path,
            metavar="DIR",
            default=Path("."),
            help="Relative or absolute path to the working directory. If none is given, the current directory is used.",
        )
        self.add_argument(
            "-ex",
            "--exclusionfile",
            type=Path,
            metavar="FILE",
            dest="exclusion_file",
            help="Path to the file that contains samplenames to be excluded.",
        )
        self.add_argument(
            "-p",
            "--prefix",
            type=str,
            metavar="PATH",
            default=None,
            help="Conda or singularity prefix. Basically a path to the place where you want to store the conda environments or the singularity images.",
        )
        self.add_argument(
            "-l",
            "--local",
            action="store_true",
            help="If this flag is present, the pipeline will be run locally (not attempting to send the jobs to an HPC cluster**). The default is to assume that you are working on a cluster. **Note that currently only LSF clusters are supported.",
        )
        self.add_argument(
            "-tl",
            "--time-limit",
            type=int,
            metavar="INT",
            default=60,
            help="Time limit per job in minutes (passed as -W argument to bsub). Jobs will be killed if not finished in this time.",
        )
        self.add_argument(
            "-u",
            "--unlock",
            action="store_true",
            help="Unlock output directory (passed to snakemake).",
        )
        self.add_argument(
            "-n",
            "--dryrun",
            action="store_true",
            help="Dry run printing steps to be taken in the pipeline without actually running it (passed to snakemake).",
        )
        self.add_argument(
            "-q",
            "--queue",
            type=str,
            default="bio",
            help="Name of the queue that the job will be submitted to if working on a cluster.",
        )
        self.add_argument(
            "--no-containers",
            action="store_false",
            dest="use_singularity",
            help="Use conda environments instead of containers.",
        )
        self.add_argument(
            "--snakemake-args",
            nargs="*",
            default={},
            action=SnakemakeKwargsAction,
            help="Extra arguments to be passed to snakemake API (https://snakemake.readthedocs.io/en/stable/api_reference/snakemake.html).",
        )

    def _parse_args(self) -> argparse.Namespace:
        """Function to parse args from the self.parser object and set
        parameters.

        It sets pipeline parameters, updates snakemake arguments
        configs. If you choose to override the behaviour in a inheriting
        class, be sure to call this original function with
        super()._parse_args().
        """
        ### Parse args and set relevant properties
        args = self.parser.parse_args(self.argv)

        self.snakemake_args.update(args.snakemake_args)
        self.local: bool = args.local
        self.unlock: bool = args.unlock
        self.dryrun: bool = args.dryrun
        self.time_limit: int = args.time_limit
        self.queue: str = args.queue

        self.workdir: Path = args.workdir.resolve()
        self.input_dir: Path = args.input.resolve()
        self.output_dir: Path = args.output.resolve()
        self.path_to_audit = self.output_dir.joinpath("audit_trail").resolve()
        self.snakemake_report = self.path_to_audit.joinpath("snakemake_report.html")
        self.snakemake_config["input_dir"] = str(self.input_dir)
        self.snakemake_config["output_dir"] = str(self.output_dir)
        self.snakemake_args["use_singularity"] = args.use_singularity
        self.snakemake_args["use_conda"] = not args.use_singularity

        if self.snakemake_args["use_conda"]:
            self.snakemake_args["conda_prefix"] = args.prefix
        if self.snakemake_args["use_singularity"]:
            self.snakemake_args["singularity_prefix"] = args.prefix

        try:
            self.exclusion_file = args.exclusion_file.resolve()
        except AttributeError:
            pass  # No exclusion file given on the command line

        return args

    def __check_input_dir(self, expected_files_dirs: List[str]) -> bool:
        """
        Check whether the input directory contains the expected files and/or directories.

        Parameters
        ----------
        expected_files_dirs : List[str]
            List of patterns that are expected to be found in the input directory.

        Returns
        -------
        bool
            True if all expected files and/or directories are found in the input directory.

        Notes
        -----
        Per pattern, multiple hits can be expected (e.g. multiple cgMLST output directories in one run).
        This function does not distinguish between files and directories.

        """
        checks_all_patterns = [False] * len(expected_files_dirs)
        for i, file_or_dir_pattern in enumerate(expected_files_dirs):
            nr_files_this_pattern = len(list(self.input_dir.glob(file_or_dir_pattern)))
            checks_all_patterns[i] = nr_files_this_pattern > 0
        return all(checks_all_patterns)

    def __parse_input_type(self) -> None:
        """
        Convert self.input_type to a tuple if it is a string.

        This function can be deprecated when all pipelines have switched to using a tuple for self.input_type.

        """
        conversion_dict = {
            "fastq": ("fastq",),
            "fasta": ("fasta",),
            "vcf": ("vcf",),
            "bam": ("bam",),
            "both": ("fastq", "fasta"),
            "fastq_and_fasta": ("fastq", "fasta"),
            "fastq_and_vcf": ("fastq", "vcf"),
            "bam_and_vcf": ("bam", "vcf"),
        }
        # check if self.input_type is a str or a tuple
        if isinstance(self.input_type, str):
            self.input_type = tuple(conversion_dict[self.input_type])

    def __build_sample_dict(self) -> None:
        """Look for samples in input_dir and set self.sample_dict accordingly.

        It also checks whether the input_dir is an output dir if
        juno_assembly and sets self.input_dir_is_juno_assembly_output.
        """
        self.sample_dict: dict[str, dict[str, str]] = {}
        self.input_dir_is_juno_assembly_output = self.__check_input_dir(
            ["clean_fastq", "de_novo_assembly_filtered"]
        )
        self.input_dir_is_juno_mapping_output = self.__check_input_dir(
            ["mapped_reads/duprem", "variants", "reference/reference.fasta"]
        )
        self.input_dir_is_juno_variant_typing_output = self.__check_input_dir(
            ["*/consensus", "audit_trail"]
        )
        self.input_dir_is_juno_cgmlst_output = self.__check_input_dir(
            ["cgmlst/*", "audit_trail"]
        )
        if self.input_dir_is_juno_assembly_output:
            self.__enlist_fastq_samples(self.input_dir.joinpath("clean_fastq"))
            self.__enlist_samples_custom_extension(
                self.input_dir.joinpath("de_novo_assembly_filtered"),
                extension=".fasta",
                key="assembly",
            )
        elif self.input_dir_is_juno_mapping_output:
            self.__enlist_samples_custom_extension(
                self.input_dir.joinpath("mapped_reads", "duprem"),
                extension=".bam",
                key="bam",
            )
            self.__enlist_samples_custom_extension(
                self.input_dir.joinpath("variants"), extension=".vcf", key="vcf"
            )
            self.__enlist_reference(self.input_dir)
        elif self.input_dir_is_juno_variant_typing_output:
            consensus_paths = list(self.input_dir.glob("*/consensus"))
            assert len(consensus_paths) == 1, error_formatter(
                f"""Expected to find exactly one consensus directory in the input directory ({self.input_dir}).\n
                Found {len(list(consensus_paths))}."""
            )
            self.__enlist_samples_custom_extension(
                consensus_paths[0], extension=".fasta", key="assembly"
            )
        elif self.input_dir_is_juno_cgmlst_output:
            raise NotImplementedError(
                "Using juno-cgmlst output is not yet implemented."
            )
            # TODO: juno-cgmlst should output a TSV file with the cgmlst results per sample, which should be enlisted here. Can be multiple schemes per sample.
            # Could be in the format: {sample: {cgmlst_scheme1: cgmlst_file1, cgmlst_scheme2: cgmlst_file2}}
            # self.__enlist_samples_custom_extension(self.input_dir.joinpath("cgmlst"), extension=".tsv", key="cgmlst")
        else:
            self.__parse_input_type()  # TODO: remove this line when self.input_type is a list in all pipelines
            if "fastq" in self.input_type:
                self.__enlist_fastq_samples(self.input_dir)
            if "fasta" in self.input_type:
                self.__enlist_samples_custom_extension(
                    self.input_dir, extension=".fasta", key="assembly"
                )
            if "vcf" in self.input_type:
                self.__enlist_samples_custom_extension(
                    self.input_dir, extension=".vcf", key="vcf"
                )
                self.__enlist_reference(self.input_dir)
            if "bam" in self.input_type:
                self.__enlist_samples_custom_extension(
                    self.input_dir, extension=".bam", key="bam"
                )

    def __enlist_fastq_samples(self, dir: Path) -> None:
        """Function to enlist the fastq files found in the input directory.
        File with too little lines are silently ignored. Adds or updates
        self.sample_dict with the form:

        {sample: {R1: fastq_file1, R2: fastq_file2}}
        """
        # Regex to detect different sample names in de fastq file names
        # It does NOT accept sample names that contain _1 or _2 in the name
        # because they get confused with the identifiers of forward and reverse
        # reads.
        # TODO: add functionality to enlist samples with only one fastq file for ONT sequencing
        pattern = re.compile(
            r"(.*?)(?:_S\d+_|_)(?:L\d{3}_)?(?:p)?R?(1|2)(?:_.*|\..*)?\.f(ast)?q(\.gz)?"
        )
        observed_combinations: Dict[Tuple[str, str], str] = {}
        errors = []
        for file_ in dir.iterdir():
            filepath_ = str(file_.resolve())
            if validate_file_has_min_lines(file_, self.min_num_lines):
                if match := pattern.fullmatch(file_.name):
                    sample_name = match.group(1)
                    read_group = match.group(2)
                    if sample_name in self.excluded_samples:
                        continue
                    # check if sample_name and read_group combination is already seen before
                    # if this happens, it might be that the sample is spread over multiple sequencing lanes
                    if (sample_name, read_group) in observed_combinations:
                        observed_file = observed_combinations[sample_name, read_group]
                        errors.append(
                            KeyError(
                                f"Multiple fastq files ({observed_file} and {filepath_}) matching the same sample ({sample_name}) and read group ({read_group}). This pipeline expects only one fastq file per sample and read group."
                            )
                        )
                    else:
                        observed_combinations[(sample_name, read_group)] = filepath_
                    sample = self.sample_dict.setdefault(match.group(1), {})
                    sample[f"R{read_group}"] = filepath_
        if len(errors) == 1:
            raise errors[0]
        elif len(errors) > 1:
            raise KeyError(errors)

    def __enlist_reference(self, dir: Path) -> None:
        ref_path = dir.joinpath("reference", "reference.fasta")
        for sample in self.sample_dict:
            if "reference" not in self.sample_dict[sample]:
                self.sample_dict[sample]["reference"] = str(ref_path.resolve())

    def __enlist_samples_custom_extension(
        self, dir: Path, extension: str, key: str
    ) -> None:
        """Function to enlist files found in the input directory based on a custom extension.
        Adds or updates self.sample_dict with the form:

        {sample: {key: file.extension}}
        """
        pattern = re.compile(f"(.*?){extension}")
        for file_ in dir.iterdir():
            if validate_file_has_min_lines(file_, self.min_num_lines):
                if match := pattern.fullmatch(file_.name):
                    sample_name = match.group(1)
                    if sample_name in self.excluded_samples:
                        continue
                    sample = self.sample_dict.setdefault(sample_name, {})
                    sample[key] = str(file_.resolve())

    def __set_exluded_samples(self) -> None:
        """Read self.exclusion file and set self.excluded_sameples.

        Function to exclude e.g. low quality samples that are specified
        by the user in a .txt file, given in the argument parser with
        the option -ex or --exclude.
        """
        if self.exclusion_file:
            with open(self.exclusion_file, "r") as f:
                self.excluded_samples: set[str] = {
                    x.replace("\n", "") for x in f.readlines()
                }

    def __validate_sample_dict(self) -> bool:
        """Validate self.sample_dict.

        Checks whether for every sample found there is a fastq and/or fasta.

        Raises:
            ValueError: if self.sample_dict does not exist
            errors: For each sample that can not be found an error is raised
            KeyError: Containing all errors that are found

        Returns:
            bool: True if everythin is fine
        """
        if not self.sample_dict:
            raise ValueError(
                error_formatter(
                    f"The input directory ({self.input_dir}) does not contain any files with the expected format/naming. Also check that your files have an expected size (min. number of lines expected: {self.min_num_lines})"
                )
            )
        errors = []
        if "fastq" in self.input_type:
            for sample in self.sample_dict:
                R1_present = "R1" in self.sample_dict[sample].keys()
                R2_present = "R2" in self.sample_dict[sample].keys()
                if not R1_present or not R2_present:
                    errors.append(
                        KeyError(
                            f"One of the paired fastq files (R1 or R2) are missing for sample {sample}. This pipeline ONLY ACCEPTS PAIRED READS. If you are sure you have complete paired-end reads, make sure to NOT USE _1 and _2 within your file names unless it is to differentiate paired fastq files or any unsupported character (Supported: letters, numbers, underscores)."
                        )
                    )
        if "fasta" in self.input_type:
            for sample in self.sample_dict:
                assembly_present = self.sample_dict[sample].keys()
                if "assembly" not in assembly_present:
                    errors.append(
                        KeyError(
                            f"The assembly is missing for sample {sample}. This pipeline expects an assembly per sample."
                        )
                    )
        if "vcf" in self.input_type:
            for sample in self.sample_dict:
                vcf_present = self.sample_dict[sample].keys()
                if "vcf" not in vcf_present:
                    errors.append(
                        KeyError(
                            f"The VCF file is missing for sample {sample}. This pipeline expects a VCF per sample."
                        )
                    )
        if "bam" in self.input_type:
            for sample in self.sample_dict:
                bam_present = self.sample_dict[sample].keys()
                if "bam" not in bam_present:
                    errors.append(
                        KeyError(
                            f"The BAM file is missing for sample {sample}. This pipeline expects a BAM per sample."
                        )
                    )
        if len(errors) == 0:
            return True
        if len(errors) == 1:
            raise errors[0]
        else:
            raise KeyError(errors)

    def get_metadata_from_csv_file(
        self,
        filepath: Optional[Path] = None,
        expected_colnames: list[str] = ["sample", "genus"],
    ) -> None:
        """Expects csv with metadata per sample, sets self.juno_metadata dict.

        Args:
            filepath (Optional[Path], optional): The location of the csv. Defaults to None.
            expected_colnames (list[str], optional): The expected header of the csv. Defaults to ["sample", "genus"].
        """
        if not filepath:
            # Only when the input_dir comes from the Juno-assembly pipeline
            # the input directory will have a sub-directory called identify_species
            # containing a top1_species_multireport.csv file that can be used as the
            # metadata for other downstream pipelines
            juno_species_file = self.input_dir.joinpath(
                "identify_species", "top1_species_multireport.csv"
            )
        else:
            juno_species_file = filepath.resolve()
        if juno_species_file.exists():
            sample_metadata = read_csv(juno_species_file, dtype={"sample": str})
            assert all(
                [col in sample_metadata.columns for col in expected_colnames]
            ), error_formatter(
                f'The provided metadata file ({filepath}) does not contain one or more of the expected column names ({",".join(expected_colnames)}). Are you using the right capitalization for the column names?'
            )
            sample_metadata.set_index("sample", inplace=True)
            self.juno_metadata = cast(
                Dict[str, Any], sample_metadata.to_dict(orient="index")
            )

    def __write_git_audit_file(self, git_file: Path) -> None:
        """Function to get URL and commit from pipeline repo.

        Args:
            git_file (Path): The file that the info is written to.
        """

        git_audit = {"repo": get_repo_url("."), "commit": get_commit_git(".")}
        with open(git_file, "w") as file:
            yaml.dump(git_audit, file, default_flow_style=False)

    def _write_pipeline_audit_file(self, pipeline_file: Path) -> None:
        """Get the pipeline_info and print it to a file for audit trail."""
        pipeline_info = {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.date_and_time,
            "hostname": self.hostname,
            "run_id": self.unique_id,
        }
        with open(pipeline_file, "w") as file:
            yaml.dump(pipeline_info, file, default_flow_style=False)

    def __write_conda_audit_file(self, conda_file: Path) -> None:
        """Get list of environments in current conda environment."""
        conda_audit = subprocess.check_output(["conda", "list"]).strip().decode("utf-8")
        with open(conda_file, "w") as file:
            file.writelines("Master environment list:\n\n")
            file.write(str(conda_audit))

    def _generate_audit_trail(self) -> list[Path]:
        """Produce audit trail in the output_dir.

        Most file contents are produced within this function but the
        sample_sheet and the user_parameters file should be produced in
        the individual pipelines and this step just ensures a copy is
        stored in the output_dir for audit trail
        """
        self.path_to_audit.mkdir(parents=True, exist_ok=True)
        print(message_formatter(f"Making audit trail in {str(self.path_to_audit)}."))

        assert pathlib.Path(
            self.sample_sheet
        ).exists(), f"The sample sheet ({str(self.sample_sheet)}) does not exist. Either this file was not created properly by the pipeline or was deleted before starting the pipeline."
        assert pathlib.Path(
            self.user_parameters_file
        ).exists(), f"The provided user_parameters ({self.user_parameters_file}) does not exist. Either this file was not created properly by the pipeline or was deleted before starting the pipeline"

        git_file = self.path_to_audit.joinpath("log_git.yaml")
        self.__write_git_audit_file(git_file)

        conda_file = self.path_to_audit.joinpath("log_conda.txt")
        self.__write_conda_audit_file(conda_file)

        pipeline_file = self.path_to_audit.joinpath("log_pipeline.yaml")
        self._write_pipeline_audit_file(pipeline_file)

        if self.exclusion_file is not None:
            shutil.copy(self.exclusion_file, self.path_to_audit)

        user_parameters_audit_file = self.path_to_audit.joinpath("user_parameters.yaml")
        subprocess.run(
            ["cp", self.user_parameters_file, user_parameters_audit_file],
            check=True,
            timeout=60,
        )
        samples_audit_file = self.path_to_audit.joinpath("sample_sheet.yaml")
        subprocess.run(
            ["cp", self.sample_sheet, samples_audit_file], check=True, timeout=60
        )
        return [
            git_file,
            conda_file,
            pipeline_file,
            user_parameters_audit_file,
            samples_audit_file,
        ]

    def _make_snakemake_report(self) -> bool:
        """Function to make a snakemake report after having run a pipeline.

        Note that it expects that the output files were already produced
        by the run_snakemake function
        """
        print(message_formatter(f"Generating snakemake report for audit trail..."))
        # The copy of the sample sheet that was generated for audit trail is
        # used instead of the original sample sheet. This is to avoid that if
        # a new run is started while there is one running, the correct sample
        # sheet for this new run is used.
        snakemake_report_successful: bool = snakemake(
            self.snakefile,
            workdir=self.workdir,
            config=self.snakemake_config,
            configfiles=[self.user_parameters_file],
            report=str(self.snakemake_report),
        )
        return snakemake_report_successful
