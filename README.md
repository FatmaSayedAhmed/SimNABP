# SimNABP (A Simple Deep Learning Model for Nanobody-Antigen Binding Prediction)
In vertebrates, antibody-mediated immunity is a crucial immune system component. Antibodies are a fast-expanding class of medicinal products. The variable domains of heavy chain-only antibodies (VHH), also called nanobodies (Nbs), are a unique class of antibodies that have recently emerged as a stable and reasonably priced substitute for full-length antibodies. These monomeric antigen-binding domains are derived from camelid heavy chain-only antibodies. Small size, strong target selectivity, notable solubility, and stability are characteristics of Nbs that promote the creation of high-caliber drugs.

With the first nanobody medication being approved in 2018, the use of nanobodies is growing quickly. However, the fact that most antigens do not have available nobodies presents a significant obstacle in developing nanobodies. Improving the binding affinities and specificities of nanobodies and antigens requires understanding their interactions. The experimental detection of nanobody antigen interactions is an important but often expensive and time-consuming stage in developing nanobody therapies. While various computational techniques have been developed to screen possible nanobodies, their reliance on 3D structures still restricts their applicability. This research aims to build a deep-learning model that can predict nanobody-antigen binding from sequence information alone. Antigens and nanobodies were each encoded using a unique combination of amino acids in a simple convolutional neural network design. The suggested model performed adequately on a carefully curated independent test dataset, with an Area Under the Curve (AUC) of 0.94.

# File Description 
- Sequence_Processing&EmbeddingMethods.ipynb

  A Jupyter notebook that includes:
    1. Prepare the binding and non-binding nanobody-antigen pair sequences.
    2. Transform the nanobody-antigen pair sequences into word sequences (1-mer and 3-mers), marking a word as "unk" if it contains any letter outside the standard 20 amino acids.
    3. Create a tokenizer for 1-mer and 3-mer amino acids (AAs) with 20^1+1 and 20^3+2 words, respectively.
    4. Convert 1-mer and 3-mers AAs to integer indexes according to the corresponding tokenizer (each word has its own unique index), and save these encoded indexes as .npz files to use as input to the model.
    5. Create and save a numpy file (.npy) for each embedding method (One-Hot, BLOSUM, and Prot2vec), which is used as the embedding weights for the model's embedding layer.
     
- train_test_split.py

  It is used to split the nanobody-antigen pairs dataset into 70% training set and 30% test set, and save the split sets to use for each experiment. 
    
- models.py

  It contains the implementation of our proposed SimNABP model


- train.py

  Perform model training using the One-Hot, BLOSUM, or Prot2Vec embeddings.

- test.py

  Evaluate the performance of the model.
  
You can find a sample of the weight of the model mentioned in our paper on google Drive: https://drive.google.com/drive/folders/1owWSzVKDcCj967HxOJdgaLeXkHY7ObmT?usp=sharing
