# Gene Pathway Extraction

1. Run the notebook named load_reader.ipynb, the scripts in reader.py are attached and will get embed_label.csv and SENtences.tsv files under ./data directory The all_the_paper.npz file can be downloaded from https://drive.google.com/file/d/1zAXvgXgCbMdIEARNy7HzmnUXgHLhKlHN/view?usp=sharing

2. Clone the fastText model to ./model directory and run the following:

$ gitclone https://github.com/facebookresearch/fastText.git

$ cd fastText

$ make

$ ./fasttext skipgram -input all_the_paper.txt/fil9 -output model/fil9

3. Download the Pre-trained word vectors https://s3-uswest-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip from https://fasttext.cc/docs/en/englishvectors.html

4. Unzip this wiki-news-300d-1M.vec.zip file and upload the wiki.en.bin and fil9.bin to ./model directory

5. Upload SENtences.tsv and relation_embed.tsv to http://projector.tensorflow.org/ can help to visualize our sentence embedding result.

6. Then run the Embedding_and_K-means.ipynb file to generate the relationships embedding and K-mean clustering visualization result.

7. Run the [main.ipynb](./main.ipynb) to replicate the results.


## This is the pipeline of TextCNN approach
![image](https://github.com/Sapphirine/Gene-Pathway-Extraction/blob/master/figures/TextCNN.png?raw=true)
  
  
## Connected Graph for known lung cancer genes in the existing archived lung cancer gene pathways
![image](https://github.com/Sapphirine/Gene-Pathway-Extraction/blob/master/figures/known_cancer_genes.png?raw=true)
  
  
## Connected Graph for genes that are related to known lung cancer genes.
![image](https://github.com/Sapphirine/Gene-Pathway-Extraction/blob/master/figures/related_genes.png?raw=true)
  
  
## Connected Graph for all genes present in the corpus.
![image](https://github.com/Sapphirine/Gene-Pathway-Extraction/blob/master/figures/all_genes.png?raw=true)
      
