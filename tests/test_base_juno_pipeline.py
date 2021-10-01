import argparse
import os
import pathlib
from sys import path
import subprocess
import unittest

main_script_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
path.insert(0, main_script_path)
from base_juno_pipeline import base_juno_pipeline
from base_juno_pipeline import helper_functions

def make_non_empty_file(file_path, content='this\nfile\nhas\ncontents'):
    with open(file_path, 'w') as file_:
        file_.write(content)

class TestTextJunoHelpers(unittest.TestCase):
    """Testing Text Helper Functions"""
    
    def test_error_formatter(self):
        """Testing that the error formatter does add the color codes to the text"""
        JunoHelpers = helper_functions.JunoHelpers()
        self.assertEqual(JunoHelpers.error_formatter('message'),
                        '\033[0;31mmessage\n\033[0;0m')
    
    def test_message_formatter(self):
        """Testing that the message formatter does add the color codes to the text"""
        JunoHelpers = helper_functions.JunoHelpers()
        self.assertEqual(JunoHelpers.message_formatter('message'),
                        '\033[0;33mmessage\n\033[0;0m')


class TestFileJunoHelpers(unittest.TestCase):
    """Testing File Helper Functions"""

    def test_validate_file_has_min_lines(self):
        """Testing that the function to check whether a file is empty works.
        It should return True if nonempty file"""
        JunoHelpers = helper_functions.JunoHelpers()
        nonempty_file = 'nonempty.txt'
        make_non_empty_file(nonempty_file)
        self.assertTrue(JunoHelpers.validate_file_has_min_lines(nonempty_file))
        os.system(f'rm -f {nonempty_file}')
    
    def test_validate_file_has_min_lines(self):
        """Testing that the function to check whether a file is empty works.
        It should return True if file is empty but no min_num_lines is given
        and False if empty file and a min_num_lines of at least 1"""
        JunoHelpers = helper_functions.JunoHelpers()
        empty_file = 'empty.txt'
        open(empty_file, 'a').close()
        self.assertFalse(
            JunoHelpers.validate_file_has_min_lines(empty_file, min_num_lines=1)
            )
        os.system(f'rm -f {empty_file}')


    def test_validate_is_nonempty_when_gzipped(self):
        """Testing that the function to check whether a gzipped file is empty 
        works. It should return True if file is empty but no min_num_lines is 
        given and False if empty file and a min_num_lines of at least 1"""
        JunoHelpers = helper_functions.JunoHelpers()
        empty_file = 'empty.txt'
        open(empty_file, 'a').close()
        os.system(f'gzip -f {empty_file}')
        self.assertTrue(
            JunoHelpers.validate_file_has_min_lines(f'{empty_file}.gz')
            )
        self.assertFalse(
            JunoHelpers.validate_file_has_min_lines(f'{empty_file}.gz', min_num_lines = 3)
            )
        os.system(f'rm -f {empty_file}')
        os.system(f'rm -f {empty_file}.gz')


class TestTextJunoHelpers(unittest.TestCase):
    """Testing Helper Functions"""

    @unittest.skipIf(not pathlib.Path('/data/BioGrid/hernanda/').exists(),
                    "Skipped in GitHub Actions because it unexplicably (so far) fails there")
    def test_git_url_of_base_juno_pipeline(self):
        """Testing if the git URL is retrieved properly (taking this folder
        as example
        """
        try:
            is_repo = subprocess.check_output(['git', 'rev-parse', '--git-dir'], 
                                                cwd=main_script_path)
        except:
            self.skipTest("The directory containint this package is not a git repo, therefore test was skipped")
        
        JunoHelpers = helper_functions.JunoHelpers()
        url = JunoHelpers.get_repo_url(main_script_path)
        url_is_correct = url == 'https://github.com/RIVM-bioinformatics/base_juno_pipeline.git'
        self.assertTrue(url_is_correct)

    def test_fail_when_dir_not_repo(self):
        """Testing that the url is 'not available' when the directory is not
        a git repo
        """
        JunoHelpers = helper_functions.JunoHelpers()
        self.assertEqual(JunoHelpers.get_repo_url(os.path.expanduser('~')),
                        'Not available. This might be because this folder is not a repository or it was downloaded manually instead of through the command line.')
    
    @unittest.skipIf(not pathlib.Path('/data/BioGrid/hernanda/').exists(),
                    "Skipped in GitHub Actions because it unexplicably (so far) fails there")
    def test_get_commit_git_from_repo(self):
        """Testing that the git commit function works"""
        try:
            is_repo = subprocess.check_output(['git', 'rev-parse', '--git-dir'], 
                                                cwd=main_script_path)
        except:
            self.skipTest("The directory containint this package is not a git repo, therefore test was skipped")
        JunoHelpers = helper_functions.JunoHelpers()
        commit_available = JunoHelpers.get_commit_git(main_script_path) == 'Not available. This might be because this folder is not a repository or it was downloaded manually instead of through the command line.'
        self.assertIsInstance(JunoHelpers.get_commit_git(main_script_path), str)
        self.assertFalse(commit_available)

    def test_get_commit_git_from_non_git_repo(self):
        """Testing that the git commit function gives right output when no git repo"""
        JunoHelpers = helper_functions.JunoHelpers()
        self.assertIsInstance(JunoHelpers.get_commit_git(os.path.expanduser('~')), str)
        self.assertEqual(JunoHelpers.get_commit_git(os.path.expanduser('~')), 
                        'Not available. This might be because this folder is not a repository or it was downloaded manually instead of through the command line.')


class TestPipelineStartup(unittest.TestCase):
    """Testing the pipeline startup (generating dict with samples) from general
    Juno pipelines"""
    
    def setUpClass(): 
        """Making fake directories and files to test different case scenarios 
        for starting pipeline"""

        fake_dirs = ['fake_dir_empty', 
                    'fake_dir_wsamples', 
                    'fake_dir_incomplete',
                    'fake_dir_juno', 
                    'fake_dir_juno/clean_fastq', 
                    'fake_dir_juno/de_novo_assembly_filtered',
                    'fake_dir_juno/identify_species',
                    'fake_wrong_fastq_names']

        fake_files = ['fake_dir_wsamples/sample1_R1.fastq',
                    'fake_dir_wsamples/sample1_R2.fastq.gz',
                    'fake_dir_wsamples/sample2_R1_filt.fq',
                    'fake_dir_wsamples/sample2_R2_filt.fq.gz', 
                    'fake_dir_wsamples/sample1.fasta',
                    'fake_dir_wsamples/sample2.fasta',
                    'fake_dir_incomplete/sample1_R1.fastq',
                    'fake_dir_incomplete/sample1_R2.fastq.gz',
                    'fake_dir_incomplete/sample2_R1_filt.fq',
                    'fake_dir_incomplete/sample2_R2_filt.fq.gz',
                    'fake_dir_incomplete/sample2.fasta',
                    'fake_dir_juno/clean_fastq/1234_R1.fastq.gz',
                    'fake_dir_juno/clean_fastq/1234_R2.fastq.gz', 
                    'fake_dir_juno/de_novo_assembly_filtered/1234.fasta',
                    'fake_wrong_fastq_names/1234_S001_PE_R1.fastq.gz',
                    'fake_wrong_fastq_names/1234_S001_PE_R2.fastq.gz']     
                    
        for folder in fake_dirs:
            pathlib.Path(folder).mkdir(exist_ok = True)
        for file_ in fake_files:
            make_non_empty_file(file_)
        bracken_dir = pathlib.Path('fake_dir_juno').joinpath('identify_species')
        bracken_dir.mkdir(parents=True, exist_ok=True)
        bracken_multireport_path = bracken_dir.joinpath('top1_species_multireport.csv')
        bracken_multireport_content = "sample,genus,species\n1234,Salmonella,enterica\n"
        make_non_empty_file(bracken_multireport_path, content=bracken_multireport_content)

    def tearDownClass():
        """Removing fake directories/files"""

        fake_dirs = ['fake_dir_empty', 
                    'fake_dir_wsamples', 
                    'fake_dir_incomplete',
                    'fake_dir_juno', 
                    'fake_dir_juno/clean_fastq', 
                    'fake_dir_juno/de_novo_assembly_filtered',
                    'fake_dir_juno/identify_species',
                    'fake_wrong_fastq_names']

        for folder in fake_dirs:
            os.system('rm -rf {}'.format(str(folder)))

    def test_nonexisting_dir(self):
        """Testing the pipeline startup fails if the input directory does not 
        exist"""
        self.assertRaises(AssertionError, 
                            base_juno_pipeline.PipelineStartup, 
                            pathlib.Path('unexisting'), 
                            'both')

    def test_emptydir(self):
        """Testing the pipeline startup fails if the input directory does not 
        have expected files"""
        self.assertRaises(ValueError, 
                            base_juno_pipeline.PipelineStartup, 
                            pathlib.Path('fake_dir_empty'), 
                            'both')

    def test_incompletedir(self):
        """Testing the pipeline startup fails if the input directory is 
        missing some of the fasta files for the fastq files"""
        self.assertRaises(KeyError, 
                            base_juno_pipeline.PipelineStartup, 
                            pathlib.Path('fake_dir_incomplete'), 'both')

    def test_correctdir_fastq(self):
        """Testing the pipeline startup accepts fastq and fastq.gz files"""

        expected_output = {'sample1': {'R1': str(pathlib.Path('fake_dir_wsamples').joinpath('sample1_R1.fastq')), 
                                        'R2': str(pathlib.Path('fake_dir_wsamples').joinpath('sample1_R2.fastq.gz'))}, 
                            'sample2': {'R1': str(pathlib.Path('fake_dir_wsamples').joinpath('sample2_R1_filt.fq')), 
                                        'R2': str(pathlib.Path('fake_dir_wsamples').joinpath('sample2_R2_filt.fq.gz'))}}
        pipeline = base_juno_pipeline.PipelineStartup(pathlib.Path('fake_dir_wsamples'), 'fastq')
        self.assertDictEqual(pipeline.sample_dict, expected_output)
        
    def test_correctdir_fasta(self):
        """Testing the pipeline startup accepts fasta"""

        expected_output = {'sample1': {'assembly': str(pathlib.Path('fake_dir_wsamples').joinpath('sample1.fasta'))}, 
                            'sample2': {'assembly': str(pathlib.Path('fake_dir_wsamples').joinpath('sample2.fasta'))}}
        pipeline = base_juno_pipeline.PipelineStartup(pathlib.Path('fake_dir_wsamples'), 'fasta')
        self.assertDictEqual(pipeline.sample_dict, expected_output)
        
    def test_correctdir_both(self):
        """Testing the pipeline startup accepts both types"""

        expected_output = {'sample1': {'R1': str(pathlib.Path('fake_dir_wsamples').joinpath('sample1_R1.fastq')), 
                                        'R2': str(pathlib.Path('fake_dir_wsamples').joinpath('sample1_R2.fastq.gz')), 
                                        'assembly': str(pathlib.Path('fake_dir_wsamples').joinpath('sample1.fasta'))}, 
                            'sample2': {'R1': str(pathlib.Path('fake_dir_wsamples').joinpath('sample2_R1_filt.fq')), 
                                        'R2': str(pathlib.Path('fake_dir_wsamples').joinpath('sample2_R2_filt.fq.gz')), 
                                        'assembly': str(pathlib.Path('fake_dir_wsamples').joinpath('sample2.fasta'))}}
        pipeline = base_juno_pipeline.PipelineStartup(pathlib.Path('fake_dir_wsamples'), 'both')
        pipeline.get_metadata_from_juno_assembly()
        self.assertDictEqual(pipeline.sample_dict, expected_output)
        self.assertEqual(pipeline.juno_metadata, None)

    def test_files_smaller_than_minlen(self):
        """Testing the pipeline startup fails if you set a min_num_lines 
        different than 0"""

        self.assertRaises(ValueError, 
                            base_juno_pipeline.PipelineStartup, 
                            pathlib.Path('fake_dir_incomplete'), 
                            'both',
                            min_num_lines = 1000)

    def test_junodir_wnumericsamplenames(self):
        """Testing the pipeline startup converts numeric file names to 
        string"""
        
        expected_output = {'1234': {'R1': str(pathlib.Path('fake_dir_juno').joinpath('clean_fastq', '1234_R1.fastq.gz')), 
                                        'R2': str(pathlib.Path('fake_dir_juno').joinpath('clean_fastq', '1234_R2.fastq.gz')), 
                                        'assembly': str(pathlib.Path('fake_dir_juno').joinpath('de_novo_assembly_filtered', '1234.fasta'))}}
        expected_metadata = {'1234': {'genus': 'Salmonella',
                                    'species': 'enterica'}}
        pipeline = base_juno_pipeline.PipelineStartup(pathlib.Path('fake_dir_juno'), 'both')
        pipeline.get_metadata_from_juno_assembly()
        self.assertDictEqual(pipeline.sample_dict, expected_output)
        self.assertDictEqual(pipeline.juno_metadata, expected_metadata, pipeline.juno_metadata)

    def test_string_accepted_as_inputdir(self):
        """Testing the pipeline startup accepts string (not only pathlib.Path)
        as input"""
        
        expected_output = {'1234': {'R1': str(pathlib.Path('fake_dir_juno').joinpath('clean_fastq', '1234_R1.fastq.gz')), 
                                        'R2': str(pathlib.Path('fake_dir_juno').joinpath('clean_fastq/1234_R2.fastq.gz')), 
                                        'assembly': str(pathlib.Path('fake_dir_juno').joinpath('de_novo_assembly_filtered','1234.fasta'))}}      
        expected_metadata = {'1234': {'genus': 'Salmonella',
                                    'species': 'enterica'}}
        pipeline = base_juno_pipeline.PipelineStartup('fake_dir_juno', 'both')
        self.assertDictEqual(pipeline.sample_dict, expected_output)
        pipeline.get_metadata_from_juno_assembly(filepath='fake_dir_juno/identify_species/top1_species_multireport.csv')
        self.assertDictEqual(pipeline.juno_metadata, expected_metadata, pipeline.juno_metadata)

    def test_fail_with_wrong_fastq_naming(self):
        """Testing the pipeline startup fails with wrong fastq naming (name 
        contains _1_ in the sample name)"""
        self.assertRaises(KeyError, 
                            base_juno_pipeline.PipelineStartup, 
                            pathlib.Path('fake_wrong_fastq_names'), 
                            'fastq')


class TestRunSnakemake(unittest.TestCase):
    """Testing the RunSnakemake class. At least testing that it is constructed
    properly (not testing the run itself)"""

    def setUpClass():
        with open('user_parameters.yaml', 'a') as file_:
            file_.write('fake_parameter: null') 
        with open('fixed_parameters.yaml', 'a') as file_:
            file_.write('fake_parameter: null') 
        with open('sample_sheet.yaml', 'a') as file_:
            file_.write('fake_sample: null')
        os.mkdir('fake_input')
        pathlib.Path('fake_input/sample_a.fasta').touch()
        pathlib.Path('fake_input/sample_b.fasta').touch()
        pathlib.Path('fake_input/sample_c.fasta').touch()

    def tearDownClass():
        os.system('rm sample_sheet.yaml user_parameters.yaml fixed_parameters.yaml')
        os.system('rm -rf fake_output_dir')
        os.system('rm -rf fake_input')

    def test_fake_dryrun_setup(self):       
        fake_run = base_juno_pipeline.RunSnakemake(pipeline_name='fake_pipeline',
                                                    pipeline_version='0.1',
                                                    output_dir='fake_output_dir',
                                                    workdir=main_script_path,
                                                    sample_sheet='sample_sheet.yaml',
                                                    user_parameters='user_parameters.yaml',
                                                    fixed_parameters='fixed_parameters.yaml',
                                                    dryrun=True)
        audit_trail_path = pathlib.Path('fake_output_dir', 'audit_trail')
        self.assertIsInstance(fake_run.date_and_time, str)
        self.assertEqual(fake_run.workdir, pathlib.Path(main_script_path))
        self.assertFalse(pathlib.Path('fake_output_dir').is_dir())
        self.assertFalse(audit_trail_path.is_dir())
        self.assertFalse(audit_trail_path.joinpath('log_conda.txt').is_file())
        self.assertFalse(audit_trail_path.joinpath('log_git.yaml').is_file())
        self.assertFalse(audit_trail_path.joinpath('log_pipeline.yaml').is_file())
        self.assertFalse(audit_trail_path.joinpath('sample_sheet.yaml').is_file())
        self.assertFalse(audit_trail_path.joinpath('user_parameters.yaml').is_file())
            
    def test_fake_run_setup(self):       
        fake_run = base_juno_pipeline.RunSnakemake(pipeline_name='fake_pipeline',
                                                    pipeline_version='0.1',
                                                    output_dir='fake_output_dir',
                                                    workdir=main_script_path,
                                                    sample_sheet='sample_sheet.yaml',
                                                    user_parameters='user_parameters.yaml',
                                                    fixed_parameters='fixed_parameters.yaml')
        audit_trail_path = pathlib.Path('fake_output_dir', 'audit_trail')
        self.assertIsInstance(fake_run.date_and_time, str)
        self.assertEqual(fake_run.workdir, pathlib.Path(main_script_path))
        self.assertTrue(pathlib.Path('fake_output_dir').is_dir())
        self.assertTrue(audit_trail_path.is_dir())
        self.assertTrue(audit_trail_path.joinpath('log_conda.txt').is_file())
        self.assertTrue(audit_trail_path.joinpath('log_git.yaml').is_file())
        self.assertTrue(audit_trail_path.joinpath('log_pipeline.yaml').is_file())
        self.assertTrue(audit_trail_path.joinpath('sample_sheet.yaml').is_file())
        self.assertTrue(audit_trail_path.joinpath('user_parameters.yaml').is_file())

        pipeline_name_in_audit_trail = False
        pipeline_version_in_audit_trail = False
        with open(audit_trail_path.joinpath('log_pipeline.yaml'), "r") as git_pipeline_trail_file:
            for line in git_pipeline_trail_file:  
                if 'fake_pipeline' in line:
                    pipeline_name_in_audit_trail = True
                if '0.1' in line:
                    pipeline_version_in_audit_trail = True
        self.assertTrue(pipeline_name_in_audit_trail)
        self.assertTrue(pipeline_version_in_audit_trail)
        
        try:
            is_repo = pathlib.Path('/data/BioGrid/hernanda/').exists()
        except:
            is_repo = False

        if is_repo:
            repo_url_in_audit_trail = False
            with open(audit_trail_path.joinpath('log_git.yaml'), "r") as git_audit_trail_file:
                for line in git_audit_trail_file:  
                    if 'https://github.com/RIVM-bioinformatics/base_juno_pipeline.git' in line:
                        repo_url_in_audit_trail = True
            self.assertTrue(repo_url_in_audit_trail)

    def test_pipeline(self):     
        fake_run = base_juno_pipeline.RunSnakemake(pipeline_name='fake_pipeline',
                                                    pipeline_version='0.1',
                                                    output_dir='fake_output_dir',
                                                    workdir=main_script_path,
                                                    sample_sheet='sample_sheet.yaml',
                                                    user_parameters='user_parameters.yaml',
                                                    fixed_parameters='fixed_parameters.yaml',
                                                    snakefile='tests/Snakefile',
                                                    name_snakemake_report='fake_snakemake_report.html',
                                                    local=True)
        audit_trail_path = pathlib.Path('fake_output_dir', 'audit_trail')
        successful_run = fake_run.run_snakemake()
        successful_report = fake_run.make_snakemake_report()
        self.assertTrue(successful_run)
        self.assertTrue(pathlib.Path('fake_output_dir/fake_result.txt').exists())
        self.assertTrue(successful_report)
        self.assertTrue(audit_trail_path.joinpath('fake_snakemake_report.html').exists())
        

class TestKwargsClass(unittest.TestCase):
    """Testing Argparse action to store kwargs (to be passed to Snakemake)"""

    def test_kwargs_are_parsed(self):
        parser = argparse.ArgumentParser(
            description = "Testing parser"
        )
        parser.add_argument(
            "--snakemake-args",
            nargs='*',
            default={},
            action=helper_functions.SnakemakeKwargsAction,
            help="Extra arguments to be passed to snakemake API (https://snakemake.readthedocs.io/en/stable/api_reference/snakemake.html)."
        )
        args = parser.parse_args(['--snakemake-args', 'key1=value1', 'key2=value2'])
        expected_output = {"key1": "value1", "key2": "value2"}
        self.assertEqual(args.snakemake_args, expected_output, args.snakemake_args)


if __name__ == '__main__':
	unittest.main()