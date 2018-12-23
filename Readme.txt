1.Run the notebook named load_reader.ipynb, the scripts in reader.py will be attached and will get embed_label.csv and SENtences.tsv files.
2. Run Read_papers.ipynb load in all_the_papers.npz and generate the SENtences.tsv and embed_label.csv files.
3. Clone the fastText model to local and run the following:
$ gitclone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
$ ./fasttext skipgram -input all_the_paper.txt/fil9 -output model/fil9
4. Download the Pre-trained word vectors https://s3-uswest-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip from https://fasttext.cc/docs/en/englishvectors.html
5.Unzip this wiki-news-300d-1M.vec.zip file and upload the wiki.en.bin and fil9.bin to the ‘model’ directory
6. Upload SENtences.tsv and relation_embed.tsv to http://projector.tensorflow.org/ can help to visualize our sentence embedding result.
