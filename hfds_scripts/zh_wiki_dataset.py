from dataclasses import dataclass
import datasets

FEATURES = datasets.Features(
    {
        'text': datasets.Value('string')
    }
)

@dataclass
class ZHWikiConfig(datasets.BuilderConfig):

    data_path : str
    min_length : int = 10
    chunksize : int = 10 << 20

class zh_wiki(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = ZHWikiConfig

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    def _split_generators(self, dl_manager):

        if not self.config.data_path:
            raise ValueError(f"Data path must be specified, but got data_path={self.config.data_path}")

        return [datasets.SplitGenerator(name='train', gen_kwargs={"data_path": self.config.data_path})]
    
    def _generate_examples(self, data_path):
        with open(file, "r", encoding=self.config.encoding) as f:
            batch_idx = 0
            while True:
                batch = f.read(self.config.chunksize)
                if not batch:
                    break
                batch += f.readline()  # finish current line
                batch = batch.splitlines()
                batch = [i for i in batch if len(i)>self.config.min_length]
                yield batch_idx, {'text': batch}
                batch_idx += 1