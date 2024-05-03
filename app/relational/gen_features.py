from typing import Any, List, Tuple
import pandas as pd
import woodwork
import os
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, DropUniqueFeatureGenerator
import featuretools as ft
from tqdm import tqdm

def ww_init(df : pd.DataFrame, name : str = None, infer_dtypes: bool = True):
	if not infer_dtypes:
		df.ww.init(name=name)
		return
	from autogluon.common.features.types import (
		S_BOOL, S_TEXT, S_TEXT_AS_CATEGORY, S_IMAGE_PATH,
		R_CATEGORY
	)
	ag_dtype_infer = AutoMLPipelineFeatureGenerator(post_generators=[DropUniqueFeatureGenerator(1.0)], post_drop_duplicates=False)
	ag_dtype_infer.fit_transform(df)
	metadata = ag_dtype_infer.feature_metadata
	ag_features = ag_dtype_infer.feature_metadata.get_features()
	logical_types = dict()
	for ft in df.columns:
		if ft not in ag_features: continue
		raw, specials = metadata.get_feature_type_raw(ft), metadata.get_feature_types_special(ft)
		logical_type = None
		nullable = df[ft].isna().any()
		for special in specials:
			if S_BOOL == special:
				logical_type = 'BooleanNullable' if nullable else 'Boolean'
				break
			if S_TEXT_AS_CATEGORY == special:
				logical_type = 'Categorical'
				break
			if S_TEXT == special:
				logical_type = 'NaturalLanguage'
				break
			if S_IMAGE_PATH == special:
				logical_type = 'Filepath'
				break
			if 'datetime' in special:
				logical_type = 'Datetime'
		
		if logical_type is None and R_CATEGORY == raw:
			logical_type = 'Categorical'

		logical_types[ft] = logical_type
	df.ww.init(name=name, logical_types=logical_types)

				
	
	

class MultiDataFramesFeatureGenerator(object):
	def __init__(self,):
		self._dataframes = []
		self._relationships = []
		self._es = None
 
	def load_dataframes_from_dict(self, table_dict: dict, infer_dtypes: bool=True):
		# Load dataframe with default name
		print("Loading tables...")
		for name in tqdm(table_dict):
			df = table_dict[name]
			self._dataframes += [(name, df)]

		# Auto infer dtypes
		print("Infering Dtypes...")
		for name, df in tqdm(self._dataframes):
			ww_init(df, name, infer_dtypes)
		self._dataframes = [ndf[1] for ndf in self._dataframes]
 
	def load_dataframes(self, paths: List[str], infer_dtypes: bool=True):
		# Load dataframe with default name
		print("Loading tables...")
		for path in tqdm(paths):
			df = pd.read_csv(path)
			name = os.path.basename(os.path.splitext(path)[0])
			self._dataframes += [(name, df)]

		# Auto infer dtypes
		print("Infering Dtypes...")
		for name, df in tqdm(self._dataframes):
			ww_init(df, name, infer_dtypes)
		self._dataframes = [ndf[1] for ndf in self._dataframes]

	@property
	def dataframe_names(self):
		return [df.ww.name for df in self._dataframes]

	@property
	def dataframes(self):
		return self._dataframes

	@property
	def relationships(self):
		return self._relationships
	
	def rename_dataframe(self, df_id, name):
		self._dataframes[df_id].ww.name = name
	
	def set_types(self, df_id, logical_types: dict=None, semantic_tags: dict=None):
		self._dataframes[df_id].ww.set_types(
	  		logical_types=logical_types, semantic_tags=semantic_tags)
  
	def set_index(self, df_id, index: str):
		self._dataframes[df_id].ww.set_index(index)
  
	def set_time_index(self, df_id, index: str):
		self._dataframes[df_id].ww.set_time_index(index)
  
	def add_relationship(self, relationship: Tuple[str, str, str, str]):
		self._relationships.append(relationship)
	
	def remove_relationship(self, relationship_id):
		del self._relationships[relationship_id]
	
	def compile(self):
		from featuretools.entityset.entityset import _get_or_create_index
		self._es = ft.EntitySet(id = 'clients')
		for df in self._dataframes:
			if df.ww.index is None:
				print('creating index for column %s' % df.ww.name)
				index_name = df.ww.name + '_id'
				_, index, df = _get_or_create_index(index_name, True, df,)
				df.ww.init(schema=df.ww.schema, index=index_name)
			self._es.add_dataframe(df)

		self._es.add_relationships(self._relationships)
	
	def plot(self, to_file:str=None):
		self._es.plot(to_file)
	
	def generate_features(self,
		target_dataframe_name: str | None = None,
		agg_primitives: List | None = None,
		trans_primitives: List | None = None,
		max_depth: int = 2,
		features_only: bool = False,
		n_jobs: int = 1,
		chunk_size: int = 100,
		verbose: bool = False,
		**kwargs
	):
		if agg_primitives is None:
			agg_primitives = ["sum", "max", "skew", "min", "count", "percent_true", "num_unique", "mode"]
		if trans_primitives is None:
			trans_primitives = ["day", "month", "weekday", 'percentile', ]

		self.target_dataframe_name = target_dataframe_name
  
		# DFS with specified primitives
		ret = ft.dfs(entityset = self._es, target_dataframe_name = target_dataframe_name,
					trans_primitives = trans_primitives,
					agg_primitives=agg_primitives, 
					max_depth = max_depth, n_jobs = n_jobs, verbose = verbose,
					features_only=features_only, chunk_size=chunk_size,
					**kwargs)

		if features_only:
			return ret
		self.result_df, feature_names = ret
		target_df = None
		for df in self._dataframes:
			if df.ww.name == target_dataframe_name:
				target_df = df
				break

		return self.result_df