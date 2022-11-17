"""
The Juno pipeline library contains the basic classes to build a 
bacteriology genomics pipeline with the format used in the IDS-
bioinformatics group at the RIVM. All our pipelines use Snakemake.
"""

import shutil
from juno_library.helper_functions import *
from datetime import datetime
from pandas import read_csv
import pathlib
from pathlib import Path
import re
from snakemake import snakemake
import subprocess
from uuid import uuid4
import yaml
from dataclasses import dataclass


@dataclass
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
    min_num_lines: int = 0

    def __post_init__(self):
        """Constructor"""
        self.input_dir = pathlib.Path(self.input_dir)
        if self.exclusion_file:
            self.exclusion_file = pathlib.Path(self.exclusion_file)
        self.min_num_lines = int(self.min_num_lines)

        # Validate input files
        assert (
            self.input_dir.is_dir()
        ), f"The provided input directory ({str(self.input_dir)}) does not exist. Please provide an existing directory"
        assert self.input_type in [
            "fastq",
            "fasta",
            "both",
        ], "input_type to be checked can only be 'fastq', 'fasta' or 'both'"
        if self.exclusion_file is not None:
            assert self.exclusion_file.is_file()

    def start_juno_pipeline(self):
        """
        This function performs calls other functions in the class to complete
        all the necessary steps needed for a robust and well documented Juno
        pipeline. For instance, validating directories, making sample_dict
        (from which a sample_sheet can be made), etc. This is the main (and
        often only) function that is usually needed to run a Juno pipeline
        """
        self.supported_extensions = {
            "fastq": (".fastq", ".fastq.gz", ".fq", ".fq.gz"),
            "fasta": (".fasta"),
        }
        self.__subdirs_ = self.__define_input_subdirs()
        self.__validate_input_dir()
        print("Making a list of samples to be processed in this pipeline run...")
        print("Checking if samples need to be Excluded")
        self.make_sample_dict()
        if self.exclusion_file is not None:
            print("found exclusion file")
            self.__exclude_samples()
            print("new sample dict without samples that are excluded")
            print(self.sample_dict)
        print(
            "Validating that all expected input files per sample are present in the input directory..."
        )
        self.validate_sample_dict()

    def __input_dir_is_juno_assembly_output(self) -> bool:
        """
        Function to check whether the input directory is actually the output
        of the Juno_assembly pipeline. The Juno_assembly pipeline is often the
        first step for downstream analyses, so its output becomes the input
        directory of other pipelines
        """
        return (
            self.input_dir.joinpath("clean_fastq").exists()
            and self.input_dir.joinpath("de_novo_assembly_filtered").exists()
        )

    def __define_input_subdirs(self) -> dict[str, Path]:
        """
        Function to check whether the input is from the Juno assembly
        pipeline or just a simple input directory
        """
        # Only when the input_dir comes from the Juno-assembly pipeline
        # the fastq and fasta files do not need to be in the same
        # folder (fastq are then expected in a subfolder called
        # <input_dir>/clean_fastq and the fasta assembly files are
        # expected in a subfolder called <input_dir>/de_novo_assembly_filtered)
        if self.__input_dir_is_juno_assembly_output():
            return {
                "fastq": self.input_dir.joinpath("clean_fastq"),
                "fasta": self.input_dir.joinpath("de_novo_assembly_filtered"),
            }
        else:
            return {"fastq": self.input_dir, "fasta": self.input_dir}

    def __validate_input_subdir(self, input_subdir, extension=("fasta")) -> bool:
        """Function to validate whether the subdirectories (if applicable)
        or the input directory have files that end with the expected extension"""
        for item in input_subdir.iterdir():
            if item.is_file():
                if str(item).endswith(extension):
                    return True
        raise ValueError(
            error_formatter(
                f"Input directory ({self.input_dir}) does not contain files that end with one of the expected extensions {extension}."
            )
        )

    def __validate_input_dir(self) -> bool:
        """
        Function to check that input directory is indeed an existing directory
        that contains files with the expected extension (fastq or fasta)
        """
        if self.input_type == "both":
            fastq_subdir_validated = self.__validate_input_subdir(
                self.__subdirs_["fastq"], self.supported_extensions["fastq"]
            )
            fasta_subdir_validated = self.__validate_input_subdir(
                self.__subdirs_["fasta"], self.supported_extensions["fasta"]
            )
            return fastq_subdir_validated and fasta_subdir_validated
        else:
            return self.__validate_input_subdir(
                self.__subdirs_[self.input_type],
                self.supported_extensions[self.input_type],
            )

    def __enlist_fastq_samples(self) -> dict[str, dict[str, str]]:
        """
        Function to enlist the fastq files found in the input directory.
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
        samples: dict[str, dict[str, str]] = {}
        for file_ in self.__subdirs_["fastq"].iterdir():
            if validate_file_has_min_lines(file_, self.min_num_lines):
                match = pattern.fullmatch(file_.name)
                if match:
                    sample = samples.setdefault(match.group(1), {})
                    sample[f"R{match.group(2)}"] = str(file_)
        return samples

    def __enlist_fasta_samples(self) -> dict[str, dict[str, str]]:
        """
        Function to enlist the fasta files found in the input
        directory. Returns a dictionary with the form
        {sample: {assembly: fasta_file}}
        """
        pattern = re.compile("(.*?).fasta")
        samples: dict[str, dict[str, str]] = {}
        for file_ in self.__subdirs_["fasta"].iterdir():
            if validate_file_has_min_lines(file_, self.min_num_lines):
                match = pattern.fullmatch(file_.name)
                if match:
                    sample = samples.setdefault(match.group(1), {})
                    sample["assembly"] = str(file_)
        return samples

    def make_sample_dict(self) -> dict[str, dict[str, str]]:
        """
        Function to make a sample sheet from the input directory (expecting
        either fastq or fasta files as input)
        """
        if self.input_type == "fastq":
            samples = self.__enlist_fastq_samples()
        elif self.input_type == "fasta":
            samples = self.__enlist_fasta_samples()
        else:
            samples = self.__enlist_fastq_samples()
            samples_fasta = self.__enlist_fasta_samples()
            for k in samples.keys():
                samples[k]["assembly"] = samples_fasta[k]["assembly"]
        self.sample_dict = samples
        return samples

    def __exclude_samples(self):
        """Function to exclude low quality samples that are specified by the user in a .txt file, given in the argument
        parser with the option -ex or --exclude. Returns a sample dict as made in the function make_sample_dict"""
        if self.exclusion_file:
            exclude_file_name = self.exclusion_file.name
            print("Exclude file path: ", self.exclusion_file)
            print("Exclude file name: ", exclude_file_name)
            if self.exclusion_file.is_file() and str(self.exclusion_file).endswith(
                ".exclude"
            ):
                with open(self.exclusion_file, "r") as exclude_file_open:
                    exclude_samples = exclude_file_open.readlines()
                    print("samples to exclude: ", exclude_samples)
                    exclude_samples_stripped = [
                        x.replace("\n", "") for x in exclude_samples
                    ]
                    self.sample_dict = {
                        sample: self.sample_dict[sample]
                        for sample in self.sample_dict
                        if sample not in exclude_samples_stripped
                    }

    def validate_sample_dict(self):
        if not self.sample_dict:
            raise ValueError(
                error_formatter(
                    f"The input directory ({self.input_dir}) does not contain any files with the expected format/naming. Also check that your files have an expected size (min. number of lines expected: {self.min_num_lines})"
                )
            )
        if self.input_type == "fastq" or self.input_type == "both":
            for sample in self.sample_dict:
                R1_present = "R1" in self.sample_dict[sample].keys()
                R2_present = "R2" in self.sample_dict[sample].keys()
                if not R1_present or not R2_present:
                    raise KeyError(
                        error_formatter(
                            f"One of the paired fastq files (R1 or R2) are missing for sample {sample}. This pipeline ONLY ACCEPTS PAIRED READS. If you are sure you have complete paired-end reads, make sure to NOT USE _1 and _2 within your file names unless it is to differentiate paired fastq files or any unsupported character (Supported: letters, numbers, underscores)."
                        )
                    )
        if self.input_type == "fasta" or self.input_type == "both":
            for sample in self.sample_dict:
                assembly_present = self.sample_dict[sample].keys()
                if "assembly" not in assembly_present:
                    raise KeyError(
                        error_formatter(
                            f"The assembly is mising for sample {sample}. This pipeline expects an assembly per sample."
                        )
                    )

    def get_metadata_from_csv_file(
        self,
        filepath: Path | None = None,
        expected_colnames: list[str] = ["sample", "genus"],
    ):
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
            juno_species_file = Path(filepath)
        if juno_species_file.exists():
            juno_metadata = read_csv(juno_species_file, dtype={"sample": str})
            assert all(
                [col in juno_metadata.columns for col in expected_colnames]
            ), error_formatter(
                f'The provided metadata file ({filepath}) does not contain one or more of the expected column names ({",".join(expected_colnames)}). Are you using the right capitalization for the column names?'
            )
            juno_metadata.set_index("sample", inplace=True)
            self.juno_metadata = juno_metadata.to_dict(orient="index")


@dataclass
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
    exclusion_file: None | Path
    sample_sheet: Path = pathlib.Path("config/sample_sheet.yaml")
    user_parameters: Path = pathlib.Path("config/user_parameters.yaml")
    fixed_parameters: Path = pathlib.Path("config/pipeline_parameters.yaml")
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

    def __post_init__(self, **kwargs):
        """Constructor"""
        self.kwargs = kwargs
        self.path_to_audit = self.output_dir.joinpath("audit_trail")
        self.output_dir = Path(self.output_dir)
        self.workdir = Path(self.workdir)
        self.fixed_parameters = Path(self.fixed_parameters)
        self.snakemake_report = str(
            self.path_to_audit.joinpath(self.name_snakemake_report)
        )
        if self.exclusion_file is None:
            print("There is no exclude file")
        else:
            print("found exclude file")
            self.exclusion_file = pathlib.Path(self.exclusion_file)
            print("exclude file = ", self.exclusion_file)

    def get_run_info(self):
        """
        Produce info specific for the current run, like a random ID, timestamp
        and host name.
        """
        self.unique_id = uuid4()
        self.date_and_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.hostname = str(subprocess.check_output(["hostname"]).strip())

    def copy_exclusion_file(self):
        """
        Make a copy of the exclude file.
        """
        print("exclude file: ", self.exclusion_file)
        if self.exclusion_file is not None:
            print("Make copy of exclude file")
            shutil.copy(self.exclusion_file, self.path_to_audit)

    def get_git_audit(self, git_file: Path):
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

    def get_pipeline_audit(self, pipeline_file: Path):
        """Get the pipeline_info and print it to a file for audit trail"""
        print(
            message_formatter(
                f"Collecting information about the pipeline (see {str(pipeline_file)})"
            )
        )
        self.get_run_info()
        pipeline_info = {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.date_and_time,
            "hostname": self.hostname,
            "run_id": self.unique_id,
        }
        with open(pipeline_file, "w") as file:
            yaml.dump(pipeline_info, file, default_flow_style=False)

    def get_conda_audit(self, conda_file: Path):
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

    def generate_audit_trail(self):
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
        self.get_git_audit(git_file)

        conda_file = self.path_to_audit.joinpath("log_conda.txt")
        self.get_conda_audit(conda_file)

        pipeline_file = self.path_to_audit.joinpath("log_pipeline.yaml")
        self.get_pipeline_audit(pipeline_file)

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

    def run_snakemake(self):
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
            self.audit_trail = self.generate_audit_trail()
            self.exclusion_file = self.copy_exclusion_file()

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

        pipeline_run_successful = snakemake(
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

    def make_snakemake_report(self):
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
        snakemake_report_successful = snakemake(
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
