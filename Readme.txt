1. Run Read_papers.ipynb load in all_the_papers.npz and generate the SENtences.tsv and embed_label.csv files.
2. Clone the fastText model to local and run the following:
$ gitclone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
$ ./fasttext skipgram -input all_the_paper.txt/fil9 -output model/fil9
3. Download the Pre-trained word vectors https://s3-uswest-
1.amazonaws.com/fasttext-vectors/wiki-news-300d-
1M.vec.zip from https://fasttext.cc/docs/en/englishvectors.
html
Unzip this wiki-news-300d-1M.vec.zip file and upload the
wiki.en.bin and fil9.bin to the ‘model’ directory
4. Upload SENtences.tsv and relation_embed.tsv to
http://projector.tensorflow.org/ can help to visualize the
embedding result.
We used http://projector.tensorflow.org/ to visualize our
embedding, each sentence with four hundred
dimensions was represented by a node in the t-SNE plot. We
need to upload the embedding matrix and the corresponding
sentences.
Use selenium package and chrome drive to access to the
final url.
The other packages we are using are pdfminer, numpy,
pandas, os, io, nltk,gensim, multiprocessing, tensorflow,
keras, sklearn,urllib,time, BeautifulSoup, re, requests,
ggplot.
After you have downloaded everything into the correct
directory, you can just run the ‘main.ipynb’ to replicate our
results.
