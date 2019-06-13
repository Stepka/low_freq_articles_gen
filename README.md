# Usage

articles_generator is a module including class ArticleGenerator. All work should bdone with that class. 

After cloning code from repository need to load gpt-2 models:

```bash
pip3 install -r requirements.txt

python3 download_model.py 117M
python3 download_model.py 345M
```

There are two ways to run script:

- via single method

```python
from articles_generator import ArticleGenerator
default_path = ""
a_gen = ArticleGenerator(default_path=default_path, verbose=1)
a_gen.process_all_steps()
```

- via sequence steps (useful for colab to see intermediate dataframes)

```python
from articles_generator import ArticleGenerator
default_path = ""
a_gen = ArticleGenerator(default_path=default_path, verbose=1)
a_gen.step_load_data()
a_gen.step_prepare_tf_hub()
a_gen.step_clusterize_questions()
a_gen.step_generate_questions_texts()
a_gen.step_merge_texts_to_questions()
a_gen.step_load_texts()
a_gen.step_extract_sentences()
a_gen.step_find_closest_sentences_to_question()
a_gen.save_articles()
```

# Result
Final processed file store to `/articles_by_question.csv`

# Configuration

There are several configuration constants in the `ArticleGenerator`:


- `ArticleGenerator.MAX_CLUSTER_SIZE`: 
Maximum unique questions that can be in the cluster while questions clusterziation. Default is `250`

- `ArticleGenerator.BASE_DBSCAN_EPS`: 
For questions clusterziation.
If less then items in cluster will be closer. It is start value for clusterziation. Default is `0.4`

- `ArticleGenerator.DBSCAN_EPS_MULT_STEP`: 
For questions clusterziation.
If found cluster size bigger then `MAX_CLUSTER_SIZE`, 
clusterization will launch next cycle with new `epsilon = self.BASE_DBSCAN_EPS/multiplicator`. 
At start `multiplicator = 1` and with each new cycle it decreased by `DBSCAN_EPS_MULT_STEP`. 
Default is `0.3`

- `ArticleGenerator.BATCH_SIZE`: 
Batch size for Embedding module. Default is `1000`

- `ArticleGenerator.NUM_CLOSEST_SENTENCES`: 
Number of the closest sentences to question selected from all. Use on STEP 8. Default is `20`

## Special thanks to
### GPT-2 library
See [DEVELOPERS.md](./DEVELOPERS.md)

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)


```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```