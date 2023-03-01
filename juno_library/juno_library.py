"""
The Juno pipeline library contains the basic classes to build a 
bacteriology genomics pipeline with the format used in the IDS-
bioinformatics group at the RIVM. All our pipelines use Snakemake.
"""

import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from pandas import read_csv
from snakemake import snakemake

from juno_library.helper_functions import *
from juno_library.version import *
from typing import Any


@dataclass(kw_only=True)
class PipelineStartup:
    """
    Class to perform actions that need to be done before running a pipeline.
    For instance: check that input directory exists and has the expected input
    files, generate a dictionary (sample_dict) with sample names and their
    corresponding files, make a dictionary with metadata if necessary.
    It has been written to be adapted to different pipelines accepting fastq and/or fasta files
    as input
    """

    input_dir: Path
    input_type: str = "both"
    exclusion_file: None | Path = None
    excluded_samples: set[str] = field(default_factory=set)
    min_num_lines: int = 0
    fasta_dir: None | Path = None
    fastq_dir: None | Path = None

    def __post_init__(self) -> None:
        """
        Validates the input directory and builds a sample_dict.
        This function is not automatically run if the class is inherited.
        """
        # Convert str to Path if needed
        self.input_dir = pathlib.Path(self.input_dir).resolve()

        assert self.input_type in [
            "fastq",
            "fasta",
            "both",
        ], "input_type to be checked can only be 'fastq', 'fasta' or 'both'"

        # Check if an exclusion file is given
        if self.exclusion_file:
            self.exclusion_file = self.exclusion_file.resolve()
            self.__set_exluded_samples()
            assert self.exclusion_file.is_file()

        # Validate input files
        assert (
            self.input_dir.is_dir()
        ), f"The provided input directory ({str(self.input_dir)}) does not exist. Please provide an existing directory"

        # Check if the input directory is created by juno-assembly and properly set up sample_dict if so
        print(
            message_formatter(
                "Making a list of samples to be processed in this pipeline run..."
            )
        )
        self.__build_sample_dict()
        print(
            message_formatter(
                "Validating that all expected input files per sample are present in the input directory..."
            )
        )
        self.__validate_sample_dict()

    def __build_sample_dict(self) -> None:
        """
        Looks for samples in the input_dir and sets self.sample_dict accordingly.
        It also checks whether the input_dir is an output dir if juno_assembly.
        """
        self.sample_dict: dict[str, dict[str, str]] = {}
        self.input_dir_is_juno_assembly_output = (
            self.input_dir.joinpath("clean_fastq").exists()
            and self.input_dir.joinpath("de_novo_assembly_filtered").exists()
        )
        if self.input_dir_is_juno_assembly_output:
            self.__enlist_fastq_samples(self.input_dir.joinpath("clean_fastq"))
            self.__enlist_fasta_samples(
                self.input_dir.joinpath("de_novo_assembly_filtered")
            )
        else:
            if self.input_type in ["fastq", "both"]:
                self.__enlist_fastq_samples(self.input_dir)
            if self.input_type in ["fasta", "both"]:
                self.__enlist_fasta_samples(self.input_dir)

    def __enlist_fastq_samples(self, dir: Path) -> None:
        """
        Function to enlist the fastq files found in the input directory.
        File with too little lines are silently ignored.
        Returns a dictionary with the form:
        {sample: {R1: fastq_file1, R2: fastq_file2}}
        """
        # Regex to detect different sample names in de fastq file names
        # It does NOT accept sample names that contain _1 or _2 in the name
        # because they get confused with the identifiers of forward and reverse
        # reads.
        pattern = re.compile(
            r"(.*?)(?:_S\d+_|_S\d+.|_|\.)(?:_L555_)?(?:p)?R?(1|2)(?:_.*\.|\..*\.|\.)f(ast)?q(\.gz)?"
        )
        for file_ in dir.iterdir():
            if validate_file_has_min_lines(file_, self.min_num_lines):
                if match := pattern.fullmatch(file_.name):
                    sample_name = match.group(1)
                    read_group = match.group(2)
                    if sample_name in self.excluded_samples:
                        continue
                    sample = self.sample_dict.setdefault(match.group(1), {})
                    sample[f"R{read_group}"] = str(file_.resolve())

    def __enlist_fasta_samples(self, dir: Path) -> None:
        """
        Function to enlist the fasta files found in the input
        directory. Returns a dictionary with the form
        {sample: {assembly: fasta_file}}
        """
        pattern = re.compile("(.*?).fasta")
        for file_ in dir.iterdir():
            if validate_file_has_min_lines(file_, self.min_num_lines):
                if match := pattern.fullmatch(file_.name):
                    sample_name = match.group(1)
                    if sample_name in self.excluded_samples:
                        continue
                    sample = self.sample_dict.setdefault(sample_name, {})
                    sample["assembly"] = str(file_.resolve())

    def __set_exluded_samples(self) -> None:
        """Function to exclude low quality samples that are specified by the user in a .txt file, given in the argument
        parser with the option -ex or --exclude. Returns a sample dict as made in the function make_sample_dict
        """
        if self.exclusion_file:
            if not self.exclusion_file.is_file() and str(self.exclusion_file).endswith(
                ".exclude"
            ):
                print(
                    error_formatter(
                        f"Exclusion file:\n\t{self.exclusion_file}\ndoes not exist or does not end with '.exclude'. It is ignored."
                    )
                )
            with open(self.exclusion_file, "r") as exclude_file_open:
                exclude_samples = exclude_file_open.readlines()
                self.excluded_samples = {x.replace("\n", "") for x in exclude_samples}

    def __validate_sample_dict(self) -> bool:
        if not self.sample_dict:
            raise ValueError(
                error_formatter(
                    f"The input directory ({self.input_dir}) does not contain any files with the expected format/naming. Also check that your files have an expected size (min. number of lines expected: {self.min_num_lines})"
                )
            )
        errors = []
        if self.input_type in ["fastq", "both"]:
            for sample in self.sample_dict:
                R1_present = "R1" in self.sample_dict[sample].keys()
                R2_present = "R2" in self.sample_dict[sample].keys()
                if not R1_present or not R2_present:
                    errors.append(
                        KeyError(
                            f"One of the paired fastq files (R1 or R2) are missing for sample {sample}. This pipeline ONLY ACCEPTS PAIRED READS. If you are sure you have complete paired-end reads, make sure to NOT USE _1 and _2 within your file names unless it is to differentiate paired fastq files or any unsupported character (Supported: letters, numbers, underscores)."
                        )
                    )
        if self.input_type in ["fasta", "both"]:
            for sample in self.sample_dict:
                assembly_present = self.sample_dict[sample].keys()
                if "assembly" not in assembly_present:
                    errors.append(
                        KeyError(
                            f"The assembly is missing for sample {sample}. This pipeline expects an assembly per sample."
                        )
                    )
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise KeyError(errors)
        return True

    def get_metadata_from_csv_file(
        self,
        filepath: Path | None = None,
        expected_colnames: list[str] = ["sample", "genus"],
    ) -> None:
        """
        Function to get a dictionary with the sample, genus and species per
        sample
        """
        if filepath is None:
            # Only when the input_dir comes from the Juno-assembly pipeline
            # the input directory will have a sub-directory called identify_species
            # containing a top1_species_multireport.csv file that can be used as the
            # metadata for other downstream pipelines
            juno_species_file = self.input_dir.joinpath(
                "identify_species", "top1_species_multireport.csv"
            )
        else:
            juno_species_file = Path(filepath).resolve()
        if juno_species_file.exists():
            juno_metadata = read_csv(juno_species_file, dtype={"sample": str})
            assert all(
                [col in juno_metadata.columns for col in expected_colnames]
            ), error_formatter(
                f'The provided metadata file ({filepath}) does not contain one or more of the expected column names ({",".join(expected_colnames)}). Are you using the right capitalization for the column names?'
            )
            juno_metadata.set_index("sample", inplace=True)
            self.juno_metadata = juno_metadata.to_dict(orient="index")
        else:
            juno_metadata = None


@dataclass(kw_only=True)
class RunSnakemake:
    """
    Class with necessary input to actually run Snakemake. It is basically a
    wrapper for the snakemake function (of the snakemake package) but with some
    customization that is used in all our Juno pipelines
    """

    pipeline_name: str
    pipeline_version: str
    output_dir: Path
    workdir: Path
    exclusion_file: None | Path = None
    sample_sheet: Path = pathlib.Path("config/sample_sheet.yaml").resolve()
    user_parameters: Path = pathlib.Path("config/user_parameters.yaml").resolve()
    fixed_parameters: Path = pathlib.Path("config/pipeline_parameters.yaml").resolve()
    snakefile: str = "Snakefile"
    cores: int = 300
    local: bool = False
    queue: str = "bio"
    unlock: bool = False
    rerunincomplete: bool = True
    dryrun: bool = False
    useconda: bool = True
    conda_prefix: None | str = None
    usesingularity: bool = True
    singularityargs: str = ""
    singularity_prefix: str | None = None
    restarttimes: int = 0
    latency_wait: int = 60
    time_limit: int = 60
    name_snakemake_report: str = "snakemake_report.html"
    conda_frontend: str = "mamba"
    unique_id: UUID = uuid4()
    date_and_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    hostname = str(subprocess.check_output(["hostname"]).strip())
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, **kwargs: Any) -> None:
        """Constructor"""
        self.kwargs = kwargs
        self.path_to_audit = self.output_dir.joinpath("audit_trail")
        self.output_dir = Path(self.output_dir).resolve()
        self.workdir = Path(self.workdir).resolve()
        self.fixed_parameters = Path(self.fixed_parameters)
        self.snakemake_report = str(
            self.path_to_audit.joinpath(self.name_snakemake_report)
        )

    def __copy_exclusion_file_to_audit_path(self) -> None:
        """
        Make a copy of the exclude file.
        """
        if self.exclusion_file is not None:
            shutil.copy(self.exclusion_file, self.path_to_audit)

    def write_git_audit_file(self, git_file: Path) -> None:
        """
        Function to get URL and commit from pipeline repo (if downloaded
        through git)
        """
        print(
            message_formatter(
                f"Collecting information about the Git repository of this pipeline (see {str(git_file)})"
            )
        )
        git_audit = {"repo": get_repo_url("."), "commit": get_commit_git(".")}
        with open(git_file, "w") as file:
            yaml.dump(git_audit, file, default_flow_style=False)

    def write_pipeline_audit_file(self, pipeline_file: Path) -> None:
        """Get the pipeline_info and print it to a file for audit trail"""
        print(
            message_formatter(
                f"Collecting information about the pipeline (see {str(pipeline_file)})"
            )
        )

        pipeline_info = {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.date_and_time,
            "hostname": self.hostname,
            "run_id": self.unique_id,
        }
        with open(pipeline_file, "w") as file:
            yaml.dump(pipeline_info, file, default_flow_style=False)

    def write_conda_audit_file(self, conda_file: Path) -> None:
        """
        Get list of environments in current conda environment
        """
        print(
            message_formatter(
                "Getting information of the master environment used for this pipeline."
            )
        )
        conda_audit = subprocess.check_output(["conda", "list"]).strip()
        with open(conda_file, "w") as file:
            file.writelines("Master environment list:\n\n")
            file.write(str(conda_audit))

    def generate_audit_trail(self) -> list[Path]:
        """
        Produce audit trail in the output_dir. Most file contents are produced
        within this function but the sample_sheet and the user_parameters file
        should be produced in the individual pipelines and this step just
        ensures a copy is stored in the output_dir for audit trail
        """
        assert pathlib.Path(
            self.sample_sheet
        ).exists(), f"The sample sheet ({str(self.sample_sheet)}) does not exist. Either this file was not created properly by the pipeline or was deleted before starting the pipeline."
        assert pathlib.Path(
            self.user_parameters
        ).exists(), f"The provided user_parameters ({self.user_parameters}) does not exist. Either this file was not created properly by the pipeline or was deleted before starting the pipeline"

        git_file = self.path_to_audit.joinpath("log_git.yaml")
        self.write_git_audit_file(git_file)

        conda_file = self.path_to_audit.joinpath("log_conda.txt")
        self.write_conda_audit_file(conda_file)

        pipeline_file = self.path_to_audit.joinpath("log_pipeline.yaml")
        self.write_pipeline_audit_file(pipeline_file)

        user_parameters_audit_file = self.path_to_audit.joinpath("user_parameters.yaml")
        subprocess.run(
            ["cp", self.user_parameters, user_parameters_audit_file],
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

    def run_snakemake(self) -> bool:
        """
        Main function to run snakemake. It has all the pre-determined input for
        running a Juno pipeline. Everything is customizable to run outside the
        RIVM but the defaults are set for the RIVM and especially for an LSF
        cluster. The commands are for now set to use bsub so it will not work
        with other types of clusters but it is on the to-do list to do it.
        """
        print(message_formatter(f"Running {self.pipeline_name} pipeline."))

        # Generate pipeline audit trail only if not dryrun (or unlock)
        # store the exclusion file in the audit_trail as well
        if not self.dryrun or self.unlock:
            self.path_to_audit.mkdir(parents=True, exist_ok=True)
            self.audit_trail_files = self.generate_audit_trail()
            self.__copy_exclusion_file_to_audit_path()

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
                'bsub -q %s \
                    -n {threads} \
                    -o %s/{name}_{wildcards}_{jobid}.out \
                    -e %s/{name}_{wildcards}_{jobid}.err \
                    -R "span[hosts=1]" \
                    -R "rusage[mem={resources.mem_gb}G]" \
                    -M {resources.mem_gb}G \
                    -W %s '
                % (
                    str(self.queue),
                    str(cluster_log_dir),
                    str(cluster_log_dir),
                    str(self.time_limit),
                )
            )

        pipeline_run_successful: bool = snakemake.snakemake(
            self.snakefile,
            workdir=self.workdir,
            configfiles=[self.user_parameters, self.fixed_parameters],
            config={"sample_sheet": str(self.sample_sheet)},
            cores=self.cores,
            nodes=self.cores,
            cluster=cluster,
            jobname=self.pipeline_name + "_{name}.jobid{jobid}",
            use_conda=self.useconda,
            conda_frontend=self.conda_frontend,
            conda_prefix=self.conda_prefix,
            use_singularity=self.usesingularity,
            singularity_args=self.singularityargs,
            singularity_prefix=self.singularity_prefix,
            keepgoing=True,
            printshellcmds=True,
            force_incomplete=self.rerunincomplete,
            restart_times=self.restarttimes,
            latency_wait=self.latency_wait,
            unlock=self.unlock,
            dryrun=self.dryrun,
            **self.kwargs,
        )
        assert pipeline_run_successful, error_formatter(
            f"An error occured while running the {self.pipeline_name} pipeline."
        )
        print(message_formatter(f"Finished running {self.pipeline_name} pipeline!"))
        return pipeline_run_successful

    def make_snakemake_report(self) -> bool:
        """
        Function to make a snakemake report after having run a pipeline. Note
        that it expects that the output files were already produced by the
        run_snakemake function
        """
        print(message_formatter(f"Generating snakemake report for audit trail..."))
        # The copy of the sample sheet that was generated for audit trail is
        # used instead of the original sample sheet. This is to avoid that if
        # a new run is started while there is one running, the correct sample
        # sheet for this new run is used.
        snakemake_report_successful: bool = snakemake.snakemake(
            self.snakefile,
            workdir=self.workdir,
            configfiles=[self.user_parameters, self.fixed_parameters],
            config={
                "sample_sheet": str(self.path_to_audit.joinpath("sample_sheet.yaml"))
            },
            cores=1,
            nodes=1,
            use_conda=self.useconda,
            conda_frontend=self.conda_frontend,
            conda_prefix=self.conda_prefix,
            use_singularity=self.usesingularity,
            singularity_args=self.singularityargs,
            singularity_prefix=self.singularity_prefix,
            report=self.snakemake_report,
            **self.kwargs,
        )
        return snakemake_report_successful
