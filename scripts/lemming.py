#!/usr/bin/env python3
import argparse
import csv
import itertools
import operator
from pathlib import Path, PosixPath
import sys
import tempfile
from typing import IO

import sh
from termcolor import cprint

from eval_tagging_accuracy import tagging_accuracy

class ExistingFilePath(PosixPath):
	"""A subclass of path that is guaranteed to point to an existing file"""
	def __new__(cls, *args, **kwargs):
		obj = super().__new__(cls, *args, **kwargs)
		if not obj.is_file():
			raise TypeError(obj)
			# raise FileNotFoundError(obj)
		return obj

class ExistingDirectoryPath(PosixPath):
	"""A subclass of path that is guaranteed to point to an existing directory"""
	def __new__(cls, *args, **kwargs):
		obj = super().__new__(cls, *args, **kwargs)
		if not obj.is_dir():
			raise TypeError(obj)
			# raise NotADirectoryError(obj)
		return obj

def parse_args() -> argparse.Namespace:
	# NOTE: DEFAULT_MARMOT_JAR was converted to str due to the way that argparse works: it will not apply the `type` function
	# to the default data unless [the default is a string type](https://docs.python.org/3/library/argparse.html#default)
	DEFAULT_MARMOT_JAR = str(Path(__file__).resolve().parent.parent / 'lib' / 'cistern' / 'marmot' / 'marmot.jar')
	class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
		pass

	parser = argparse.ArgumentParser(description='lemmingatizer: A python wrapper around the open-source package `Lemming` for joint lemmatization and POS/morphological tagging.')
	parser.set_defaults(subcommand=None)
	subparsers = parser.add_subparsers()

	###### TRAIN ####
	train = subparsers.add_parser('train', formatter_class=Formatter, help='Train a MarMoT and Lemming model to learn to tag and lemmatize surface form sequences.')
	train.set_defaults(subcommand='train')

	dirs = train.add_argument_group('Directories')
	dirs.add_argument('--ud_data_dir', type=ExistingDirectoryPath, required=True, help='Directory containing train & dev CONLLU files')
	dirs.add_argument('--exp_dir', type=Path, required=True, help='Directory in which to generate output files')

	marmot = train.add_argument_group('MarMoT setup', description='Setup for MarMoT')
	marmot.add_argument('--marmot_jar', type=ExistingFilePath, default=DEFAULT_MARMOT_JAR, help='Path to JAR file to use with MarMoT')
	marmot.add_argument('--java_heap_limit', type=int, default=20, help='Java heap limit size in GB')

	optionals = train.add_argument_group('Optional MarMoT Training Features', description='Optional Features for the tagger/lemmatizer')
	optionals.add_argument('--embedding', '-e', type=ExistingFilePath, help='Optional path to the type embedding file for use by MarMoT')
	optionals.add_argument('--train_token_limit', type=int, default=None, help='The number of lines to truncate train to, or not set if using full train')

	continuations = train.add_argument_group('Continuations')
	continuations.add_argument('--annotate', type=bool, default=False, help='Whether to compute annotate after training')
	continuations.add_argument('--accuracy', type=bool, default=False, help='Whether to compute accuracy after annotation')

	columns = train.add_argument_group('CONLLU Columns', description='Column indices within CONLLU formatted files corresponding to each data type')
	columns.add_argument('--form_column', default=1, type=int, help='Index of inflected surface form')
	columns.add_argument('--lemma_column', default=2, type=int, help='Index of lemma')
	columns.add_argument('--tag_column', default=3, type=int, help='Index of POS tag')
	columns.add_argument('--morph_column', default=5, type=int, help='Index of morphological tag')

	###### ANNOTATE ####
	annotate = subparsers.add_parser('annotate', formatter_class=Formatter, help='Annotate a surface form sequence with lemmata, pos tags, and morph tags.')
	annotate.set_defaults(subcommand='annotate')

	marmot = annotate.add_argument_group('MarMoT setup', description='Setup for MarMoT')
	marmot.add_argument('--marmot_jar', type=ExistingFilePath, default=DEFAULT_MARMOT_JAR, help='Path to JAR file to use with MarMoT')
	marmot.add_argument('--java_heap_limit', type=int, default=20, help='Java heap limit size in GB')
	marmot.add_argument('--marmot_model', type=ExistingFilePath, required=True, help='Path to trained MarMoT model')
	marmot.add_argument('--lemming_model', type=ExistingFilePath, required=True, help='Path to trained Lemming model')

	io = annotate.add_argument_group('File I/O')
	io.add_argument('--input_file', type=ExistingFilePath, required=True, help='Path of file to annotate')
	io.add_argument('--input_form_column', default=1, type=int, help='Index of inflected surface form in input file')
	io.add_argument('--pred_file', type=Path, required=True, help='Path to save annotated CONLLU file to')

	continuations = annotate.add_argument_group('Continuations')
	continuations.add_argument('--accuracy', type=bool, default=False, help='Whether to compute accuracy after annotation')

	columns = annotate.add_argument_group('CONLLU Columns', description='Column indices within CONLLU formatted files corresponding to each data type')
	columns.add_argument('--token_idx_column', default=0, type=int, help='Index of token indices')
	columns.add_argument('--form_column', default=1, type=int, help='Index of inflected surface form')
	columns.add_argument('--lemma_column', default=2, type=int, help='Index of lemma')
	columns.add_argument('--tag_column', default=3, type=int, help='Index of POS tag')
	columns.add_argument('--morph_column', default=5, type=int, help='Index of morphological tag')

	###### ACCURACY ####
	accuracy = subparsers.add_parser(
		'accuracy', formatter_class=Formatter,
		help='Compute tagging/lemmatization accuracy of MarMoT/Lemming predictions against an oracle.',
		description='Prints tagging accuracy of a prediction file versus an oracle. Accuracy can either be computed with respect to POS tags, morphological tags, lemmas, or some combination.')
	accuracy.set_defaults(subcommand='accuracy')

	optionals = accuracy.add_argument_group('Accuracy Calculation Options')
	optionals.add_argument('--tag', type=str, choices='pos morph mtag lemma'.split(), default=['pos', 'morph'], nargs='+', help='Whether to compute accuracy using POS tags, morph tags, lemmata, or some combination')
	optionals.add_argument('--only_oov', action='store_true', default=False, help='Whether to compute accuracy using using only OOV words (as determined by vocab_file')

	io = accuracy.add_argument_group('File I/O')
	io.add_argument('--oracle_file', type=ExistingFilePath, required=True, help='Path to oracle CONLLU file')
	io.add_argument('--pred_file', type=ExistingFilePath, required=True, help='Path to prediction CONLLU file')
	io.add_argument('--vocab_file', type=ExistingFilePath, help='Path to in-vocabulary CONLLU file')

	columns = accuracy.add_argument_group('CONLLU Columns', description='Column indices within CONLLU formatted files corresponding to each data type')
	columns.add_argument('--token_idx_column', default=0, type=int, help='Index of token indices')
	columns.add_argument('--form_column', default=1, type=int, help='Index of inflected surface form')
	columns.add_argument('--lemma_column', default=2, type=int, help='Index of lemma')
	columns.add_argument('--tag_column', default=3, type=int, help='Index of POS tag')
	columns.add_argument('--morph_column', default=5, type=int, help='Index of morphological tag')
	if len(sys.argv) == 1:
		parser.print_help(sys.stderr)
		sys.exit(1)
	return parser.parse_args()

not_yellow_iter = itertools.cycle('red, green, blue, magenta, cyan, white'.split(', '))

def cmd(command):
	command = sh.Command(command)
	def constructed_command(argstring):
		color = next(not_yellow_iter)
		def colored_print(data, file, **kwargs):
			cprint(data.strip(), color=color, **kwargs)
		def colored_stdout(data):
			colored_print(data, on_color='on_white', file=sys.stdout)
			# cprint(data.strip(), color=color, file=sys.stdout)
		def colored_stderr(data):
			colored_print(data, file=sys.stderr)
			# cprint(data.strip(), color=color, on_color='on_yellow', file=sys.stderr)
		
		p = None
		try:
			p = command(*argstring.split(), _out=colored_stdout, _err=colored_stderr, _bg=True)
			p.wait()
			return p
		except KeyboardInterrupt:
			if p is not None:
				p.kill()
			raise
		
	return constructed_command

def globbed_single_file_find(path: Path, glob: str) -> Path:
	try:
		return next(path.glob(glob))
	except StopIteration as e:
		raise FileNotFoundError(path / glob) from e

def open_possibly_with_gzip(filename: Path, mode: str='r', compresslevel: int=2) -> IO:
    """
    Assume the file is gzip compressed if the filename ends with `.gz`.
    Else, open as a normal text file
    """
    f = filename.open(mode)
    if '.gz' in filename.suffixes:
        import gzip
        f = gzip.open(f, compresslevel=compresslevel)
    return f

def train(args: argparse.Namespace):

	
	# find CONLLU format train and test files from UD folder
	full_train = globbed_single_file_find(args.ud_data_dir, '*-train.conllu')
	if args.annotate:
		full_dev = globbed_single_file_find(args.ud_data_dir, '*-dev.conllu')

	# if we're truncating the training data, then truncate to the first k tokens, else use full data
	should_truncate = (args.train_token_limit is not None and args.train_token_limit > 0)
	working_exp_dir = args.exp_dir / (args.train_token_limit if should_truncate else 'full')
	working_exp_dir.mkdir(parents=True, exist_ok=True)
	if should_truncate:
		trunc_train = working_exp_dir / 'train.conllu'
		cmd('python3')(f'scripts/truncate_train.py --input {full_train} --output {trunc_train} --limit {args.train_token_limit}')
		new_train = ExistingFilePath(trunc_train)
	else:
		new_train = ExistingFilePath(full_train)

	tagger_model = working_exp_dir / 'model.marmot'
	java = cmd('java')
	train_command = f'-Xmx{args.java_heap_limit}G -cp {args.marmot_jar} marmot.morph.cmd.Trainer -train-file form-index={args.form_column},lemma-index={args.lemma_column},tag-index={args.tag_column},morph-index={args.morph_column},{new_train} -beam-size 3 -tag-morph true -model-file {tagger_model} -verbose true'
	if args.embedding is not None:
		train_command += f' -type-embeddings dense=true,{args.embedding}'
	java(train_command)

	lemming_model = working_exp_dir / 'model.lemming'
	java(f'''-Xmx{args.java_heap_limit}G
		-cp
		{args.marmot_jar}:{args.marmot_jar.parent}/lib/trove-3.1a1.jar:{args.marmot_jar.parent}/lib/mallet.jar
		lemming.lemma.cmd.Trainer
		lemming.lemma.ranker.RankerTrainer
		use-morph=true,offline-feature-extraction=true,tag-dependent=true
		{lemming_model}
		form-index={args.form_column},lemma-index={args.lemma_column},tag-index={args.tag_column},morph-index={args.morph_column},{new_train}
	''')

	# if we should follow up training with annotation
	if args.annotate:
		args.marmot_model = tagger_model
		args.lemming_model = lemming_model
		args.input_form_column = args.form_column
		args.token_idx_column = 0

		args.input_file = full_train
		args.pred_file = working_exp_dir / 'train-pred.conllu'
		annotate(args)

		args.input_file = full_dev
		args.pred_file = working_exp_dir / 'dev-pred.conllu'
		annotate(args)


def annotate(args: argparse.Namespace):

	with tempfile.NamedTemporaryFile('r', dir=args.pred_file.expanduser().resolve().parent) as temp:
		temp_path = temp.name
		cmd('java')(f'-Xmx{args.java_heap_limit}G -cp {args.marmot_jar} marmot.morph.cmd.Annotator -model-file {args.marmot_model} -lemmatizer-file {args.lemming_model} -test-file form-index={args.input_form_column},{args.input_file} -pred-file {temp_path} -tag-morph true -lemmatize true -verbose true')

		# reorder columns to be in line with settings
		with open_possibly_with_gzip(args.pred_file, mode='w') as pred_file:
			col_count = 1 + max([args.form_column, args.lemma_column, args.tag_column, args.morph_column])
			temp.seek(0)
			for row in csv.reader(temp, delimiter='\t', quoting=csv.QUOTE_NONE):
				if row:
					out_row = ['_'] * col_count

					# these indices correspond to the indices produced as
					# output from the MarMoT annotator
					index = row[0]
					form = row[1]
					lemma = row[3]
					pos = row[5]
					morph = row[7]
					
					out_row[args.token_idx_column] = index
					out_row[args.form_column] = form
					out_row[args.lemma_column] = lemma
					out_row[args.tag_column] = pos
					out_row[args.morph_column] = morph

					print('\t'.join(out_row), file=pred_file)
				else:
					print(file=pred_file)

	if args.accuracy:
		args.tag = ['lemma']
		args.pred_file = args.pred_file
		args.oracle_file = args.input_file
		accuracy(args)
	# ######
	# prediction_path = working_exp_dir / 'train-pred.conllu'
	
	# dev_pred = working_exp_dir / 'dev-pred.conllu'
	# full_dev   = globbed_single_file_find(args.ud_data_dir, '*-dev.conllu')
	# 
	# lemming_annotate(f'{args.marmot_jar} {tagger_model} {lemming_model} {args.form_column} {full_dev} {dev_pred}')

def accuracy(args: argparse.Namespace):

	criteria = []
	if 'pos' in args.tag:
		pos = operator.attrgetter('pos')
		criteria.append(pos)
	if 'morph' in args.tag or 'mtag' in args.tag:
		morph = operator.attrgetter('mtag')
		criteria.append(morph)
	if 'lemma' in args.tag:
		lemma = operator.attrgetter('lemma')
		criteria.append(lambda p: lemma(p).lower())

	tag2str = lambda p: ';'.join(cr(p) for cr in criteria)

	idx_col, form_col, lem_col, pos_col, mtag_col = args.token_idx_column, args.form_column, args.lemma_column, args.tag_column, args.morph_column

	print(tagging_accuracy(args.oracle_file, args.pred_file, args.vocab_file, tag2str, idx_col, form_col, lem_col, pos_col, mtag_col, only_oov=args.only_oov))

if __name__ == '__main__':
	args = parse_args()
	if args.subcommand == 'train':
		train(args)
	elif args.subcommand == 'annotate':
		annotate(args)
	elif args.subcommand == 'accuracy':
		accuracy(args)

