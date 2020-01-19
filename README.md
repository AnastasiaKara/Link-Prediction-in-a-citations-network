Predict links in a citation network with a classification supervised approach using Apache Spark and scala language.

File descriptions:

training_set.txt: Contains 615.512 labeled node pairs (1 if there is an edge between the two nodes, 0 else).
One pair and label per row, as: source node ID, target node ID, and 1 or 0. The IDs match the papers
in the node_information.csv file.

testing_set.txt: Contains 32,648 node pairs. The file contains one node pair per row, as: source node ID,
target node ID. The labels are not available in this file, we have to find them!

node_information.csv: For each paper out of 27,770, contains the following information (1) unique
ID, (2) publication year (between 1993 and 2003), (3) title, (4) authors, (5) name of journal (not
available for all papers), and (6) abstract. Abstracts are already in lowercase, common English
stopwords have been removed, and punctuation marks have been removed except for intra-word
dashes. 

Cit-HepTh.txt: This is the complete ground truth network. You should use this file only to evaluate
your solutions with respect to the accuracy.
