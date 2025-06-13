# TableFormer: Table Structure Understanding with Transformers.  

Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, Peter Staar IBM Research {ahn,nli,mly,taa}@zurich.ibm.com  

# Abstract  

Tables organize valuable content in a concise and compact representation. This content is extremely valuable for systems such as search engines, Knowledge Graph’s, etc, since they enhance their predictive capabilities. Unfortunately, tables come in a large variety of shapes and sizes. Furthermore, they can have complex column/row-header configurations, multiline rows, different variety of separation lines, missing entries, etc. As such, the correct identification of the table-structure from an image is a nontrivial task. In this paper, we present a new table-structure identification model. The latter improves the latest end-toend deep learning model (i.e. encoder-dual-decoder from PubTabNet) in two significant ways. First, we introduce a new object detection decoder for table-cells. In this way, we can obtain the content of the table-cells from programmatic PDF’s directly from the PDF source and avoid the training of the custom OCR decoders. This architectural change leads to more accurate table-content extraction and allows us to tackle non-english tables. Second, we replace the LSTM decoders with transformer based decoders. This upgrade improves significantly the previous state-of-the-art tree-editing-distance-score (TEDS) from $91\%$ to $98.5\%$ on simple tables and from $88.7\%$ to $95\%$ on complex tables.  

# 1. Introduction  

The occurrence of tables in documents is ubiquitous. They often summarise quantitative or factual data, which is cumbersome to describe in verbose text but nevertheless extremely valuable. Unfortunately, this compact representation is often not easy to parse by machines. There are many implicit conventions used to obtain a compact table representation. For example, tables often have complex columnand row-headers in order to reduce duplicated cell content. Lines of different shapes and sizes are leveraged to separate content or indicate a tree structure. Additionally, tables can also have empty/missing table-entries or multi-row textual table-entries. Fig. 1 shows a table which presents all these issues.  

a. Picture of a table:   


<html><body><table><tr><td></td><td></td><td colspan="2">① Observer 1</td><td></td></tr><tr><td></td><td>③</td><td>benign</td><td></td><td>Total observer 2</td></tr><tr><td rowspan="3">2 Observer 2</td><td>Demign</td><td>13</td><td></td><td>15</td></tr><tr><td>ruafignant</td><td></td><td>62</td><td>62</td></tr><tr><td>Total observer 1</td><td>13</td><td>64</td><td></td></tr></table></body></html>  

# b. Red-annotation of bounding boxes, Blue-predictions by TableFormer  

# c. Structure predicted by TableFormer:  

![](images/0757a23dc211fa81bf2674667d46e510b7630e4b8654225687891406d930462b.jpg)  
Figure 1: Picture of a table with subtle, complex features such as (1) multi-column headers, (2) cell with multi-row text and (3) cells with no content. Image from PubTabNet evaluation set, filename: ‘PMC2944238 004 02’.  

<html><body><table><tr><td>0</td><td>2</td><td colspan="3">1</td></tr><tr><td>3</td><td>4 3</td><td>5</td><td>6</td><td>7</td></tr><tr><td rowspan="3">8 2</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td>13</td><td>14</td><td>15</td><td>16</td></tr><tr><td>17</td><td>18</td><td>19</td><td>20</td></tr></table></body></html>  

Recently, significant progress has been made with vision based approaches to extract tables in documents. For the sake of completeness, the issue of table extraction from documents is typically decomposed into two separate challenges, i.e. (1) finding the location of the table(s) on a document-page and (2) finding the structure of a given table in the document.  

The first problem is called table-location and has been previously addressed [30, 38, 19, 21, 23, 26, 8] with stateof-the-art object-detection networks (e.g. YOLO and later on Mask-RCNN [9]). For all practical purposes, it can be considered as a solved problem, given enough ground-truth data to train on.  

The second problem is called table-structure decomposition. The latter is a long standing problem in the community of document understanding [6, 4, 14]. Contrary to the table-location problem, there are no commonly used approaches that can easily be re-purposed to solve this problem. Lately, a set of new model-architectures has been proposed by the community to address table-structure decomposition [37, 36, 18, 20]. All these models have some weaknesses (see Sec. 2). The common denominator here is the reliance on textual features and/or the inability to provide the bounding box of each table-cell in the original image.  

In this paper, we want to address these weaknesses and present a robust table-structure decomposition algorithm. The design criteria for our model are the following. First, we want our algorithm to be language agnostic. In this way, we can obtain the structure of any table, irregardless of the language. Second, we want our algorithm to leverage as much data as possible from the original PDF document. For programmatic PDF documents, the text-cells can often be extracted much faster and with higher accuracy compared to OCR methods. Last but not least, we want to have a direct link between the table-cell and its bounding box in the image.  

To meet the design criteria listed above, we developed a new model called TableFormer and a synthetically generated table structure dataset called SynthTabNet1. In particular, our contributions in this work can be summarised as follows:  

• We propose TableFormer, a transformer based model that predicts tables structure and bounding boxes for the table content simultaneously in an end-to-end approach.   
• Across all benchmark datasets TableFormer significantly outperforms existing state-of-the-art metrics, while being much more efficient in training and inference to existing works.   
• We present SynthTabNet a synthetically generated dataset, with various appearance styles and complexity.   
• An augmented dataset based on PubTabNet [37], FinTabNet [36], and TableBank [17] with generated ground-truth for reproducibility.  

The paper is structured as follows. In Sec. 2, we give a brief overview of the current state-of-the-art. In Sec. 3, we describe the datasets on which we train. In Sec. 4, we introduce the TableFormer model-architecture and describe its results & performance in Sec. 5. As a conclusion, we describe how this new model-architecture can be re-purposed for other tasks in the computer-vision community.  

# 2. Previous work and State of the Art  

Identifying the structure of a table has been an outstanding problem in the document-parsing community, that motivates many organised public challenges [6, 4, 14]. The difficulty of the problem can be attributed to a number of factors. First, there is a large variety in the shapes and sizes of tables. Such large variety requires a flexible method. This is especially true for complex column- and row headers, which can be extremely intricate and demanding. A second factor of complexity is the lack of data with regard to table-structure. Until the publication of PubTabNet [37], there were no large datasets (i.e. $>100\mathrm{K}$ tables) that provided structure information. This happens primarily due to the fact that tables are notoriously time-consuming to annotate by hand. However, this has definitely changed in recent years with the deliverance of PubTabNet [37], FinTabNet [36], TableBank [17] etc.  

Before the rising popularity of deep neural networks, the community relied heavily on heuristic and/or statistical methods to do table structure identification [3, 7, 11, 5, 13, 28]. Although such methods work well on constrained tables [12], a more data-driven approach can be applied due to the advent of convolutional neural networks (CNNs) and the availability of large datasets. To the best-of-our knowledge, there are currently two different types of network architecture that are being pursued for state-of-the-art tablestructure identification.  

Image-to-Text networks: In this type of network, one predicts a sequence of tokens starting from an encoded image. Such sequences of tokens can be HTML table tags [37, 17] or LaTeX symbols[10]. The choice of symbols is ultimately not very important, since one can be transformed into the other. There are however subtle variations in the Image-to-Text networks. The easiest network architectures are “image-encoder $\rightarrow$ text-decoder” (IETD), similar to network architectures that try to provide captions to images [32]. In these IETD networks, one expects as output the LaTeX/HTML string of the entire table, i.e. the symbols necessary for creating the table with the content of the table. Another approach is the “image-encoder $\rightarrow$ dual decoder” (IEDD) networks. In these type of networks, one has two consecutive decoders with different purposes. The first decoder is the tag-decoder, i.e. it only produces the HTML/LaTeX tags which construct an empty table. The second content-decoder uses the encoding of the image in combination with the output encoding of each cell-tag (from the tag-decoder) to generate the textual content of each table cell. The network architecture of IEDD is certainly more elaborate, but it has the advantage that one can pre-train the tag-decoder which is constrained to the table-tags.  

In practice, both network architectures (IETD and IEDD) require an implicit, custom trained object-characterrecognition (OCR) to obtain the content of the table-cells. In the case of IETD, this OCR engine is implicit in the decoder similar to [24]. For the IEDD, the OCR is solely embedded in the content-decoder. This reliance on a custom, implicit OCR decoder is of course problematic. OCR is a well known and extremely tough problem, that often needs custom training for each individual language. However, the limited availability for non-english content in the current datasets, makes it impractical to apply the IETD and IEDD methods on tables with other languages. Additionally, OCR can be completely omitted if the tables originate from programmatic PDF documents with known positions of each cell. The latter was the inspiration for the work of this paper.  

Graph Neural networks: Graph Neural networks (GNN’s) take a radically different approach to tablestructure extraction. Note that one table cell can constitute out of multiple text-cells. To obtain the table-structure, one creates an initial graph, where each of the text-cells becomes a node in the graph similar to [33, 34, 2]. Each node is then associated with en embedding vector coming from the encoded image, its coordinates and the encoded text. Furthermore, nodes that represent adjacent text-cells are linked. Graph Convolutional Networks (GCN’s) based methods take the image as an input, but also the position of the text-cells and their content [18]. The purpose of a GCN is to transform the input graph into a new graph, which replaces the old links with new ones. The new links then represent the table-structure. With this approach, one can avoid the need to build custom OCR decoders. However, the quality of the reconstructed structure is not comparable to the current state-of-the-art [18].  

Hybrid Deep Learning-Rule-Based approach: A popular current model for table-structure identification is the use of a hybrid Deep Learning-Rule-Based approach similar to [27, 29]. In this approach, one first detects the position of the table-cells with object detection (e.g. $\mathrm{YoloVx}$ or MaskRCNN), then classifies the table into different types (from its images) and finally uses different rule-sets to obtain its table-structure. Currently, this approach achieves stateof-the-art results, but is not an end-to-end deep-learning method. As such, new rules need to be written if different types of tables are encountered.  

# 3. Datasets  

We rely on large-scale datasets such as PubTabNet [37], FinTabNet [36], and TableBank [17] datasets to train and evaluate our models. These datasets span over various appearance styles and content. We also introduce our own synthetically generated SynthTabNet dataset to fix an imbalance in the previous datasets.  

![](images/3444ba47c081c7d523c29c93b6125725bffd1e741311d527093dec0e07b4d20c.jpg)  
Figure 2: Distribution of the tables across different table dimensions in PubTabNet $^+$ FinTabNet datasets  

The PubTabNet dataset contains $509\mathrm{k}$ tables delivered as annotated PNG images. The annotations consist of the table structure represented in HTML format, the tokenized text and its bounding boxes per table cell. Fig. 1 shows the appearance style of PubTabNet. Depending on its complexity, a table is characterized as “simple” when it does not contain row spans or column spans, otherwise it is “complex”. The dataset is divided into Train and Val splits (roughly $98\%$ and $2\%$ ). The Train split consists of $54\%$ simple and $46\%$ complex tables and the Val split of $51\%$ and $49\%$ respectively. The FinTabNet dataset contains $112\mathrm{k}$ tables delivered as single-page PDF documents with mixed table structures and text content. Similarly to the PubTabNet, the annotations of FinTabNet include the table structure in HTML, the tokenized text and the bounding boxes on a table cell basis. The dataset is divided into Train, Test and Val splits $(81\%$ , $9.5\%$ , $9.5\%$ ), and each one is almost equally divided into simple and complex tables (Train: $48\%$ simple, $52\%$ complex, Test: $48\%$ simple, $52\%$ complex, Test: $53\%$ simple, $47\%$ complex). Finally the TableBank dataset consists of $145\mathrm{k\Omega}$ tables provided as JPEG images. The latter has annotations for the table structure, but only few with bounding boxes of the table cells. The entire dataset consists of simple tables and it is divided into $90\%$ Train, $3\%$ Test and $7\%$ Val splits.  

Due to the heterogeneity across the dataset formats, it was necessary to combine all available data into one homogenized dataset before we could train our models for practical purposes. Given the size of PubTabNet, we adopted its annotation format and we extracted and converted all tables as PNG images with a resolution of 72 dpi. Additionally, we have filtered out tables with extreme sizes due to small amount of such tables, and kept only those ones ranging between $1^{*}1$ and $20^{*}10$ (rows/columns).  

The availability of the bounding boxes for all table cells is essential to train our models. In order to distinguish between empty and non-empty bounding boxes, we have introduced a binary class in the annotation. Unfortunately, the original datasets either omit the bounding boxes for whole tables (e.g. TableBank) or they narrow their scope only to non-empty cells. Therefore, it was imperative to introduce a data pre-processing procedure that generates the missing bounding boxes out of the annotation information. This procedure first parses the provided table structure and calculates the dimensions of the most fine-grained grid that covers the table structure. Notice that each table cell may occupy multiple grid squares due to row or column spans. In case of PubTabNet we had to compute missing bounding boxes for $48\%$ of the simple and $69\%$ of the complex tables. Regarding FinTabNet, $68\%$ of the simple and $98\%$ of the complex tables require the generation of bounding boxes.  

As it is illustrated in Fig. 2, the table distributions from all datasets are skewed towards simpler structures with fewer number of rows/columns. Additionally, there is very limited variance in the table styles, which in case of PubTabNet and FinTabNet means one styling format for the majority of the tables. Similar limitations appear also in the type of table content, which in some cases (e.g. FinTabNet) is restricted to a certain domain. Ultimately, the lack of diversity in the training dataset damages the ability of the models to generalize well on unseen data.  

Motivated by those observations we aimed at generating a synthetic table dataset named SynthTabNet. This approach offers control over: 1) the size of the dataset, 2) the table structure, 3) the table style and 4) the type of content. The complexity of the table structure is described by the size of the table header and the table body, as well as the percentage of the table cells covered by row spans and column spans. A set of carefully designed styling templates provides the basis to build a wide range of table appearances. Lastly, the table content is generated out of a curated collection of text corpora. By controlling the size and scope of the synthetic datasets we are able to train and evaluate our models in a variety of different conditions. For example, we can first generate a highly diverse dataset to train our models and then evaluate their performance on other synthetic datasets which are focused on a specific domain.  

In this regard, we have prepared four synthetic datasets, each one containing 150k examples. The corpora to generate the table text consists of the most frequent terms appearing in PubTabNet and FinTabNet together with randomly generated text. The first two synthetic datasets have been fine-tuned to mimic the appearance of the original datasets but encompass more complicated table structures. The third one adopts a colorful appearance with high contrast and the last one contains tables with sparse content. Lastly, we have combined all synthetic datasets into one big unified synthetic dataset of 600k examples.  

Table 1: Both “Combined-Tabnet” and ”CombinedTabnet” are variations of the following: $({}^{*})$ The CombinedTabnet dataset is the processed combination of PubTabNet and Fintabnet. $(^{**})$ The combined dataset is the processed combination of PubTabNet, Fintabnet and TableBank.   


<html><body><table><tr><td></td><td>Tags</td><td>Bbox</td><td>Size</td><td>Format</td></tr><tr><td>PubTabNet</td><td></td><td></td><td>509k</td><td>PNG</td></tr><tr><td>FinTabNet</td><td></td><td></td><td>112k</td><td>PDF</td></tr><tr><td>TableBank</td><td>√</td><td>x</td><td>145k</td><td>JPEG</td></tr><tr><td>Combined-Tabnet(*)</td><td>√</td><td></td><td>400k</td><td>PNG</td></tr><tr><td>Combined(**)</td><td></td><td></td><td>500k</td><td>PNG</td></tr><tr><td>SynthTabNet</td><td></td><td></td><td>600k</td><td>PNG</td></tr></table></body></html>  

Tab. 1 summarizes the various attributes of the datasets.  

# 4. The TableFormer model  

Given the image of a table, TableFormer is able to predict: 1) a sequence of tokens that represent the structure of a table, and 2) a bounding box coupled to a subset of those tokens. The conversion of an image into a sequence of tokens is a well-known task [35, 16]. While attention is often used as an implicit method to associate each token of the sequence with a position in the original image, an explicit association between the individual table-cells and the image bounding boxes is also required.  

# 4.1. Model architecture.  

We now describe in detail the proposed method, which is composed of three main components, see Fig. 4. Our CNN Backbone Network encodes the input as a feature vector of predefined length. The input feature vector of the encoded image is passed to the Structure Decoder to produce a sequence of HTML tags that represent the structure of the table. With each prediction of an HTML standard data cell $(\mathrm{'}<\mathrm{td}>\mathrm{)}$ the hidden state of that cell is passed to the Cell BBox Decoder. As for spanning cells, such as row or column span, the tag is broken down to $\ '<\ '$ , ‘rowspan $\mathbf{\rho}_{=}\mathbf{\rho},$ or ‘colspan $\omega$ , with the number of spanning cells (attribute), and $\ '>'$ . The hidden state attached to $\ '<\ '$ is passed to the Cell BBox Decoder. A shared feed forward network (FFN) receives the hidden states from the Structure Decoder, to provide the final detection predictions of the bounding box coordinates and their classification.  

CNN Backbone Network. A ResNet-18 CNN is the backbone that receives the table image and encodes it as a vector of predefined length. The network has been modified by removing the linear and pooling layer, as we are not performing classification, and adding an adaptive pooling layer of size $28^{*}28$ . ResNet by default downsamples the image resolution by 32 and then the encoded image is provided to both the Structure Decoder, and Cell BBox Decoder.  

![](images/e66428f2cd7967c26a124429604c5b58925ea83ac9015a7fea6a04ce9234b0a1.jpg)  
Figure 3: TableFormer takes in an image of the PDF and creates bounding box and HTML structure predictions that are synchronized. The bounding boxes grabs the content from the PDF and inserts it in the structure.  

![](images/d711628d28daf7699d8739b505918e07939cbed88408100b1abb5f8d86a53a79.jpg)  
Figure 4: Given an input image of a table, the Encoder produces fixed-length features that represent the input image. The features are then passed to both the Structure Decoder and Cell BBox Decoder. During training, the Structure Decoder receives ‘tokenized tags’ of the HTML code that represent the table structure. Afterwards, a transformer encoder and decoder architecture is employed to produce features that are received by a linear layer, and the Cell BBox Decoder. The linear layer is applied to the features to predict the tags. Simultaneously, the Cell BBox Decoder selects features referring to the data cells $(^{\leftarrow}{<}\mathrm{td}{>}^{\prime},^{\leftarrow}{<}^{\prime})$ and passes them through an attention network, an MLP, and a linear layer to predict the bounding boxes.  

Structure Decoder. The transformer architecture of this component is based on the work proposed in [31]. After extensive experimentation, the Structure Decoder is modeled as a transformer encoder with two encoder layers and a transformer decoder made from a stack of 4 decoder layers that comprise mainly of multi-head attention and feed forward layers. This configuration uses fewer layers and heads in comparison to networks applied to other problems (e.g. “Scene Understanding”, “Image Captioning”), something which we relate to the simplicity of table images.  

The transformer encoder receives an encoded image from the CNN Backbone Network and refines it through a multi-head dot-product attention layer, followed by a Feed Forward Network. During training, the transformer decoder receives as input the output feature produced by the transformer encoder, and the tokenized input of the HTML ground-truth tags. Using a stack of multi-head attention layers, different aspects of the tag sequence could be inferred. This is achieved by each attention head on a layer operating in a different subspace, and then combining altogether their attention score.  

Cell BBox Decoder. Our architecture allows to simultaneously predict HTML tags and bounding boxes for each table cell without the need of a separate object detector end to end. This approach is inspired by DETR [1] which employs a Transformer Encoder, and Decoder that looks for a specific number of object queries (potential object detections). As our model utilizes a transformer architecture, the hidden state of the $\mathrm{<td>}$ and $\ '<\ '$ HTML structure tags become the object query.  

The encoding generated by the CNN Backbone Network along with the features acquired for every data cell from the Transformer Decoder are then passed to the attention network. The attention network takes both inputs and learns to provide an attention weighted encoding. This weighted attention encoding is then multiplied to the encoded image to produce a feature for each table cell. Notice that this is different than the typical object detection problem where imbalances between the number of detections and the amount of objects may exist. In our case, we know up front that the produced detections always match with the table cells in number and correspondence.  

The output features for each table cell are then fed into the feed-forward network (FFN). The FFN consists of a Multi-Layer Perceptron (3 layers with ReLU activation function) that predicts the normalized coordinates for the bounding box of each table cell. Finally, the predicted bounding boxes are classified based on whether they are empty or not using a linear layer.  

Loss Functions. We formulate a multi-task loss Eq. 2 to train our network. The Cross-Entropy loss (denoted as $l_{s}$ ) is used to train the Structure Decoder which predicts the structure tokens. As for the Cell BBox Decoder it is trained with a combination of losses denoted as $l_{b o x}$ . $l_{b o x}$ consists of the generally used $l_{1}$ loss for object detection and the IoU loss $(l_{i o u})$ to be scale invariant as explained in [25]. In comparison to DETR, we do not use the Hungarian algorithm [15] to match the predicted bounding boxes with the ground-truth boxes, as we have already achieved a one-toone match through two steps: 1) Our token input sequence is naturally ordered, therefore the hidden states of the table data cells are also in order when they are provided as input to the Cell BBox Decoder, and 2) Our bounding boxes generation mechanism (see Sec. 3) ensures a one-to-one mapping between the cell content and its bounding box for all post-processed datasets.  

The loss used to train the TableFormer can be defined as following:  

$$
\begin{array}{r}{l_{b o x}=\lambda_{i o u}l_{i o u}+\lambda_{l1}}\ {l=\lambda l_{s}+(1-\lambda)l_{b o x}}\end{array}
$$  

where $\lambda\in[0,1]$ , and $\lambda_{i o u},\lambda_{l1}\in\mathbb{R}$ are hyper-parameters.  

# 5. Experimental Results  

# 5.1. Implementation Details  

TableFormer uses ResNet-18 as the CNN Backbone Network. The input images are resized to $448^{*}448$ pixels and the feature map has a dimension of $28^{*}28$ . Additionally, we enforce the following input constraints:  

$$
\begin{array}{r}{\mathrm{Image~width~and~height}\leq1024~\mathrm{pixels}}\ {\mathrm{Structural~tags~length}\leq512~\mathrm{tokens}.}\end{array}
$$  

Although input constraints are used also by other methods, such as EDD, ours are less restrictive due to the improved runtime performance and lower memory footprint of TableFormer. This allows to utilize input samples with longer sequences and images with larger dimensions.  

The Transformer Encoder consists of two “Transformer Encoder Layers”, with an input feature size of 512, feed forward network of 1024, and 4 attention heads. As for the Transformer Decoder it is composed of four “Transformer Decoder Layers” with similar input and output dimensions as the “Transformer Encoder Layers”. Even though our model uses fewer layers and heads than the default implementation parameters, our extensive experimentation has proved this setup to be more suitable for table images. We attribute this finding to the inherent design of table images, which contain mostly lines and text, unlike the more elaborate content present in other scopes (e.g. the COCO dataset). Moreover, we have added ResNet blocks to the inputs of the Structure Decoder and Cell BBox Decoder. This prevents a decoder having a stronger influence over the learned weights which would damage the other prediction task (structure vs bounding boxes), but learn task specific weights instead. Lastly our dropout layers are set to 0.5.  

For training, TableFormer is trained with 3 Adam optimizers, each one for the CNN Backbone Network, Structure Decoder, and Cell BBox Decoder. Taking the PubTabNet as an example for our parameter set up, the initializing learning rate is 0.001 for 12 epochs with a batch size of 24, and $\lambda$ set to 0.5. Afterwards, we reduce the learning rate to 0.0001, the batch size to 18 and train for 12 more epochs or convergence.  

TableFormer is implemented with PyTorch and Torchvision libraries [22]. To speed up the inference, the image undergoes a single forward pass through the CNN Backbone Network and transformer encoder. This eliminates the overhead of generating the same features for each decoding step. Similarly, we employ a ’caching’ technique to preform faster autoregressive decoding. This is achieved by storing the features of decoded tokens so we can reuse them for each time step. Therefore, we only compute the attention for each new tag.  

# 5.2. Generalization  

TableFormer is evaluated on three major publicly available datasets of different nature to prove the generalization and effectiveness of our model. The datasets used for evaluation are the PubTabNet, FinTabNet and TableBank which stem from the scientific, financial and general domains respectively.  

We also share our baseline results on the challenging SynthTabNet dataset. Throughout our experiments, the same parameters stated in Sec. 5.1 are utilized.  

# 5.3. Datasets and Metrics  

The Tree-Edit-Distance-Based Similarity (TEDS) metric was introduced in [37]. It represents the prediction, and ground-truth as a tree structure of HTML tags. This similarity is calculated as:  

$$
\mathrm{TEDS}\left(T_{a},T_{b}\right)=1-\frac{\mathrm{EditDist}\left(T_{a},T_{b}\right)}{\mathrm{max}\left(\left|T_{a}\right|,\left|T_{b}\right|\right)}
$$  

where $T_{a}$ and $T_{b}$ represent tables in tree structure HTML format. EditDist denotes the tree-edit distance, and $|T|$ represents the number of nodes in $T$ .  

# 5.4. Quantitative Analysis  

Structure. As shown in Tab. 2, TableFormer outperforms all SOTA methods across different datasets by a large margin for predicting the table structure from an image. All the more, our model outperforms pre-trained methods. During the evaluation we do not apply any table filtering. We also provide our baseline results on the SynthTabNet dataset. It has been observed that large tables (e.g. tables that occupy half of the page or more) yield poor predictions. We attribute this issue to the image resizing during the preprocessing step, that produces downsampled images with indistinguishable features. This problem can be addressed by treating such big tables with a separate model which accepts a large input image size.  

Table 2: Structure results on PubTabNet (PTN), FinTabNet (FTN), TableBank (TB) and SynthTabNet (STN). FT: Model was trained on PubTabNet then finetuned.   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">TEDS</td></tr><tr><td>Dataset</td><td>Simple</td><td>Complex</td><td>All</td></tr><tr><td>EDD</td><td>PTN</td><td>91.1</td><td>88.7</td><td>89.9</td></tr><tr><td>GTE</td><td>PTN</td><td></td><td></td><td>93.01</td></tr><tr><td>TableFormer</td><td>PTN</td><td>98.5</td><td>95.0</td><td>96.75</td></tr><tr><td>EDD</td><td>FTN</td><td>88.4</td><td>92.08</td><td>90.6</td></tr><tr><td>GTE</td><td>FTN</td><td></td><td></td><td>87.14</td></tr><tr><td>GTE (FT)</td><td>FTN</td><td>一</td><td>一</td><td>91.02</td></tr><tr><td>TableFormer</td><td>FTN</td><td>97.5</td><td>96.0</td><td>96.8</td></tr><tr><td>EDD</td><td>TB</td><td>86.0</td><td></td><td>86.0</td></tr><tr><td>TableFormer</td><td>TB</td><td>89.6</td><td></td><td>89.6</td></tr><tr><td>TableFormer</td><td>STN</td><td>96.9</td><td>95.7</td><td>96.7</td></tr></table></body></html>  

Cell Detection. Like any object detector, our Cell BBox Detector provides bounding boxes that can be improved with post-processing during inference. We make use of the grid-like structure of tables to refine the predictions. A detailed explanation on the post-processing is available in the supplementary material. As shown in Tab. 3, we evaluate our Cell BBox Decoder accuracy for cells with a class label of ‘content’ only using the PASCAL VOC mAP metric for pre-processing and post-processing. Note that we do not have post-processing results for SynthTabNet as images are only provided. To compare the performance of our proposed approach, we’ve integrated TableFormer’s Cell BBox Decoder into EDD architecture. As mentioned previously, the Structure Decoder provides the Cell BBox Decoder with the features needed to predict the bounding box predictions. Therefore, the accuracy of the Structure Decoder directly influences the accuracy of the Cell BBox Decoder. If the Structure Decoder predicts an extra column, this will result in an extra column of predicted bounding boxes.  

Table 3: Cell Bounding Box detection results on PubTabNet, and FinTabNet. PP: Post-processing.   


<html><body><table><tr><td>Model</td><td>Dataset</td><td>mAP</td><td>mAP (PP)</td></tr><tr><td>EDD+BBoX</td><td>PubTabNet</td><td>79.2</td><td>82.7</td></tr><tr><td>TableFormer</td><td>PubTabNet</td><td>82.1</td><td>86.8</td></tr><tr><td>TableFormer</td><td>SynthTabNet</td><td>87.7</td><td>一</td></tr></table></body></html>  

Cell Content. In this section, we evaluate the entire pipeline of recovering a table with content. Here we put our approach to test by capitalizing on extracting content from the PDF cells rather than decoding from images. Tab. 4 shows the TEDs score of HTML code representing the structure of the table along with the content inserted in the data cell and compared with the ground-truth. Our method achieved a $5.3\%$ increase over the state-of-the-art, and commercial solutions. We believe our scores would be higher if the HTML ground-truth matched the extracted PDF cell content. Unfortunately, there are small discrepancies such as spacings around words or special characters with various unicode representations.  

Table 4: Results of structure with content retrieved using cell detection on PubTabNet. In all cases the input is PDF documents with cropped tables.   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="3">TEDS</td></tr><tr><td>Simple</td><td>Complex</td><td>All</td></tr><tr><td>Tabula</td><td>78.0</td><td>57.8</td><td>67.9</td></tr><tr><td>Traprange</td><td>60.8</td><td>49.9</td><td>55.4</td></tr><tr><td>Camelot</td><td>80.0</td><td>66.0</td><td>73.0</td></tr><tr><td>AcrobatPro</td><td>68.9</td><td>61.8</td><td>65.3</td></tr><tr><td>EDD</td><td>91.2</td><td>85.4</td><td>88.3</td></tr><tr><td>TableFormer</td><td>95.4</td><td>90.1</td><td>93.6</td></tr></table></body></html>  

a. Red - PDF cells, Green - predicted bounding boxes, Blue - post-processed predictions matched to PDF cells Japanese language (previously unseen by TableFormer): Example table from FinTabNet:  

<html><body><table><tr><td colspan="2"></td><td></td><td colspan="3"></td><td>参考又献</td><td></td></tr><tr><td colspan="2"></td><td></td><td>央韶</td><td>日本語</td><td></td><td>央韶</td><td>日本韶</td></tr><tr><td colspan="3"></td><td></td><td></td><td></td><td>150</td><td></td></tr><tr><td>omputatiol</td><td colspan="2">(COL1NG2002</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="2">電氮情报通信字会 2003年合大会</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="2">情報处理字会第65回全国大会（2003</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="2">第17回人工知能学会全国大会（2003</td><td>208</td><td></td><td>203</td><td></td><td></td></tr><tr><td></td><td colspan="2">自然言語処理研究会第146155回</td><td></td><td></td><td></td><td></td><td>5</td></tr><tr><td></td><td colspan="2">WW苏5收集七无篇文</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td>945</td><td>294</td><td>651</td><td>1122</td><td>955</td></tr></table></body></html>  

<html><body><table><tr><td></td></tr><tr><td>millions</td></tr><tr><td>RSUS F'SUs</td></tr><tr><td>NonvestedonJanuary 90.10 91.19 Grantec 17.44 122.4</td></tr><tr><td></td></tr><tr><td>Vested 0.5) 87.08 81.14</td></tr><tr><td>Canceledorforfeited 102.01 92.18 NonvestedonDecember 8.3 104.85 104.51</td></tr></table></body></html>  

![](images/caa7065509f799a744edc1eeb1a41efc6a3844dfedbc1ea8ce225166b71f097c.jpg)  

![](images/a620f8b7f5a223a53a1f5a559dc148d1cf509239d36337615e78e573f9f27874.jpg)  

b. Structure predicted by TableFormer, with superimposed matched PDF cell text:  

<html><body><table><tr><td></td><td></td><td colspan="2">文</td><td colspan="2">参考文献</td></tr><tr><td>出典</td><td>数</td><td>英語</td><td>日本語</td><td>英語</td><td>日本語</td></tr><tr><td>AssociationforComputational Linguistics(ACL2003)</td><td>65</td><td>65</td><td>0</td><td>150</td><td>0</td></tr><tr><td>ComputationalLinguistics（COLING2002)</td><td>140</td><td>140</td><td>0</td><td>150</td><td>0</td></tr><tr><td>電氮情報通信学会2003年合大会</td><td>150</td><td>8</td><td>142</td><td>223</td><td>147</td></tr><tr><td>情報处理学会第65回全国大会（2003）</td><td>177</td><td>1</td><td>176</td><td>150</td><td>236</td></tr><tr><td>第17回人工知能学会全国大会（2003）</td><td>208</td><td>5</td><td>203</td><td>152</td><td>244</td></tr><tr><td>自然言語处理研究会第146~155回</td><td>98</td><td>2</td><td>96</td><td>150</td><td>232</td></tr><tr><td>WW力收集儿文</td><td>107</td><td>73</td><td>34</td><td>147</td><td>96</td></tr><tr><td></td><td>計 945</td><td>294</td><td>651</td><td>1122</td><td>955</td></tr></table></body></html>  

<html><body><table><tr><td></td><td colspan="2">Shares（inmillions)</td><td colspan="2">WeightedAverageGrantDateFair Value</td></tr><tr><td></td><td>RSUs</td><td>PSUs</td><td>RSUs</td><td>PSUs</td></tr><tr><td>NonvestedonJanuary</td><td>1.1</td><td>0.3</td><td>90.10$</td><td>$91.19</td></tr><tr><td>Granted</td><td>0.5</td><td>0.1</td><td>117.44</td><td>122.41</td></tr><tr><td>Vested</td><td>(0.5)</td><td>(0.1)</td><td>87.08</td><td>81.14</td></tr><tr><td>Canceledorforfeited</td><td>(0.1)</td><td></td><td>102.01</td><td>92.18</td></tr><tr><td>NonvestedonDecember31</td><td>1.0</td><td>0.3</td><td>104.85$</td><td>$ 104.51</td></tr></table></body></html>

Text is aligned to match original for ease of viewing  

Figure 5: One of the benefits of TableFormer is that it is language agnostic, as an example, the left part of the illustration demonstrates TableFormer predictions on previously unseen language (Japanese). Additionally, we see that TableFormer is robust to variability in style and content, right side of the illustration shows the example of the TableFormer prediction from the FinTabNet dataset.  

![](images/b4551b22279369131f93d85cdcfa229c324b15b77feaf5930fccfa12b58b7585.jpg)  
Figure 6: An example of TableFormer predictions (bounding boxes and structure) from generated SynthTabNet table.  

# 5.5. Qualitative Analysis  

We showcase several visualizations for the different components of our network on various “complex” tables within datasets presented in this work in Fig. 5 and Fig. 6 As it is shown, our model is able to predict bounding boxes for all table cells, even for the empty ones. Additionally, our post-processing techniques can extract the cell content by matching the predicted bounding boxes to the PDF cells based on their overlap and spatial proximity. The left part of Fig. 5 demonstrates also the adaptability of our method to any language, as it can successfully extract Japanese text, although the training set contains only English content. We provide more visualizations including the intermediate steps in the supplementary material. Overall these illustrations justify the versatility of our method across a diverse range of table appearances and content type.  

# 6. Future Work & Conclusion  

In this paper, we presented TableFormer an end-to-end transformer based approach to predict table structures and bounding boxes of cells from an image. This approach enables us to recreate the table structure, and extract the cell content from PDF or OCR by using bounding boxes. Additionally, it provides the versatility required in real-world scenarios when dealing with various types of PDF documents, and languages. Furthermore, our method outperforms all state-of-the-arts with a wide margin. Finally, we introduce “SynthTabNet” a challenging synthetically generated dataset that reinforces missing characteristics from other datasets.  

# References  

[1] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to  

end object detection with transformers. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020, pages 213–229, Cham, 2020. Springer International Publishing. 5   
[2] Zewen Chi, Heyan Huang, Heng-Da Xu, Houjin Yu, Wanxuan Yin, and Xian-Ling Mao. Complicated table structure recognition. arXiv preprint arXiv:1908.04729, 2019. 3   
[3] Bertrand Couasnon and Aurelie Lemaitre. Recognition of Tables and Forms, pages 647–677. Springer London, London, 2014. 2   
[4] Herve Dejean, Jean-Luc Meunier, Liangcai Gao, Yilun Huang, Yu Fang, Florian Kleber, and Eva-Maria Lang. ICDAR 2019 Competition on Table Detection and Recognition (cTDaR), Apr. 2019. http://sac.founderit.com/. 2   
[5] Basilios Gatos, Dimitrios Danatsas, Ioannis Pratikakis, and Stavros J Perantonis. Automatic table detection in document images. In International Conference on Pattern Recognition and Image Analysis, pages 609–618. Springer, 2005. 2   
[6]  Max Gobel, Tamir Hassan, Ermelinda Oro, and Giorgio Orsi. Icdar 2013 table competition. In 2013 12th International Conference on Document Analysis and Recognition, pages 1449–1453, 2013. 2   
[7] EA Green and M Krishnamoorthy. Recognition of tables using table grammars. procs. In Symposium on Document Analysis and Recognition (SDAIR’95), pages 261–277. 2 [8] Khurram Azeem Hashmi, Alain Pagani, Marcus Liwicki, Didier Stricker, and Muhammad Zeshan Afzal. Castabdetectors: Cascade network for table detection in document images with recursive feature pyramid and switchable atrous convolution. Journal of Imaging, 7(10), 2021. 1 [9] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017. 1   
[10] Yelin He, X. Qi, Jiaquan Ye, Peng Gao, Yihao Chen, Bingcong Li, Xin Tang, and Rong Xiao. Pingan-vcgroup’s solution for icdar 2021 competition on scientific table image recognition to latex. ArXiv, abs/2105.01846, 2021. 2   
[11] Jianying Hu, Ramanujan S Kashi, Daniel P Lopresti, and Gordon Wilfong. Medium-independent table detection. In Document Recognition and Retrieval VII, volume 3967, pages 291–302. International Society for Optics and Photonics, 1999. 2   
[12] Matthew Hurst. A constraint-based approach to table structure derivation. In Proceedings of the Seventh International Conference on Document Analysis and Recognition - Volume 2, ICDAR $\ '_{03}$ , page 911, USA, 2003. IEEE Computer Society. 2   
[13] Thotreingam Kasar, Philippine Barlas, Sebastien Adam, Clément Chatelain, and Thierry Paquet. Learning to detect tables in scanned document images using line information. In 2013 12th International Conference on Document Analysis and Recognition, pages 1185–1189. IEEE, 2013. 2   
[14] Pratik Kayal, Mrinal Anand, Harsh Desai, and Mayank Singh. Icdar 2021 competition on scientific table image recognition to latex, 2021. 2   
[15] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955. 6   
[16] Girish Kulkarni, Visruth Premraj, Vicente Ordonez, Sagnik Dhar, Siming Li, Yejin Choi, Alexander C. Berg, and Tamara L. Berg. Babytalk: Understanding and generating simple image descriptions. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(12):2891–2903, 2013. 4   
[17] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. Tablebank: A benchmark dataset for table detection and recognition, 2019. 2, 3   
[18] Yiren Li, Zheng Huang, Junchi Yan, Yi Zhou, Fan Ye, and Xianhui Liu. Gfte: Graph-based financial table extraction. In Alberto Del Bimbo, Rita Cucchiara, Stan Sclaroff, Giovanni Maria Farinella, Tao Mei, Marco Bertini, Hugo Jair Escalante, and Roberto Vezzani, editors, Pattern Recognition. ICPR International Workshops and Challenges, pages 644–658, Cham, 2021. Springer International Publishing. 2, 3   
[19] Nikolaos Livathinos, Cesar Berrospi, Maksym Lysak, Viktor Kuropiatnyk, Ahmed Nassar, Andre Carvalho, Michele Dolfi, Christoph Auer, Kasper Dinkla, and Peter Staar. Robust pdf document conversion using recurrent neural networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(17):15137–15145, May 2021. 1   
[20] Rujiao Long, Wen Wang, Nan Xue, Feiyu Gao, Zhibo Yang, Yongpan Wang, and Gui-Song Xia. Parsing table structures in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 944–952, 2021. 2   
[21] Shubham Singh Paliwal, D Vishwanath, Rohit Rahul, Monika Sharma, and Lovekesh Vig. Tablenet: Deep learning model for end-to-end table detection and tabular data extraction from scanned document images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 128–133. IEEE, 2019. 1   
[22] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alche-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc., 2019. 6   
[23] Devashish Prasad, Ayan Gadpal, Kshitij Kapadni, Manish Visave, and Kavita Sultanpure. Cascadetabnet: An approach for end to end table detection and structure recognition from image-based documents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 572–573, 2020. 1   
[24] Shah Rukh Qasim, Hassan Mahmood, and Faisal Shafait. Rethinking table recognition using graph neural networks. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 142–147. IEEE, 2019. 3   
[25] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF Conference on  

and evaluation. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020, pages 564–580, Cham, 2020. Springer International Publishing. 2, 3, 7 [38] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: Largest dataset ever for document layout analysis. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1015–1022, 2019. 1  

Computer Vision and Pattern Recognition, pages 658–666, 2019. 6   
[26] Sebastian Schreiber, Stefan Agne, Ivo Wolf, Andreas Dengel, and Sheraz Ahmed. Deepdesrt: Deep learning for detection and structure recognition of tables in document images. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 01, pages 1162– 1167, 2017. 1   
[27] Sebastian Schreiber, Stefan Agne, Ivo Wolf, Andreas Dengel, and Sheraz Ahmed. Deepdesrt: Deep learning for detection and structure recognition of tables in document images. In 2017 14th IAPR international conference on document analysis and recognition (ICDAR), volume 1, pages 1162–1167. IEEE, 2017. 3   
[28] Faisal Shafait and Ray Smith. Table detection in heterogeneous documents. In Proceedings of the 9th IAPR International Workshop on Document Analysis Systems, pages 65– 72, 2010. 2   
[29] Shoaib Ahmed Siddiqui, Imran Ali Fateh, Syed Tahseen Raza Rizvi, Andreas Dengel, and Sheraz Ahmed. Deeptabstr: Deep learning based table structure recognition. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1403–1409. IEEE, 2019. 3   
[30] Peter W J Staar, Michele Dolfi, Christoph Auer, and Costas Bekas. Corpus conversion service: A machine learning platform to ingest documents at scale. In Proceedings of the 24th ACM SIGKDD, KDD ’18, pages 774–782, New York, NY, USA, 2018. ACM. 1   
[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 5998–6008. Curran Associates, Inc., 2017. 5   
[32] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2015. 2   
[33] Wenyuan Xue, Qingyong Li, and Dacheng Tao. Res2tim: reconstruct syntactic structures from table images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 749–755. IEEE, 2019. 3   
[34] Wenyuan Xue, Baosheng Yu, Wen Wang, Dacheng Tao, and Qingyong Li. Tgrnet: A table graph reconstruction network for table structure recognition. arXiv preprint arXiv:2106.10598, 2021. 3   
[35] Quanzeng You, Hailin Jin, Zhaowen Wang, Chen Fang, and Jiebo Luo. Image captioning with semantic attention. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4651–4659, 2016. 4   
[36] Xinyi Zheng, Doug Burdick, Lucian Popa, Peter Zhong, and Nancy Xin Ru Wang. Global table extractor (gte): A framework for joint table identification and cell structure recognition using visual context. Winter Conference for Applications in Computer Vision (WACV), 2021. 2, 3   
[37] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: Data, model,  

# TableFormer: Table Structure Understanding with Transformers Supplementary Material  

# 1. Details on the datasets  

# 1.1. Data preparation  

As a first step of our data preparation process, we have calculated statistics over the datasets across the following dimensions: (1) table size measured in the number of rows and columns, (2) complexity of the table, (3) strictness of the provided HTML structure and (4) completeness (i.e. no omitted bounding boxes). A table is considered to be simple if it does not contain row spans or column spans. Additionally, a table has a strict HTML structure if every row has the same number of columns after taking into account any row or column spans. Therefore a strict HTML structure looks always rectangular. However, HTML is a lenient encoding format, i.e. tables with rows of different sizes might still be regarded as correct due to implicit display rules. These implicit rules leave room for ambiguity, which we want to avoid. As such, we prefer to have ”strict” tables, i.e. tables where every row has exactly the same length.  

We have developed a technique that tries to derive a missing bounding box out of its neighbors. As a first step, we use the annotation data to generate the most fine-grained grid that covers the table structure. In case of strict HTML tables, all grid squares are associated with some table cell and in the presence of table spans a cell extends across multiple grid squares. When enough bounding boxes are known for a rectangular table, it is possible to compute the geometrical border lines between the grid rows and columns. Eventually this information is used to generate the missing bounding boxes. Additionally, the existence of unused grid squares indicates that the table rows have unequal number of columns and the overall structure is non-strict. The generation of missing bounding boxes for non-strict HTML tables is ambiguous and therefore quite challenging. Thus, we have decided to simply discard those tables. In case of PubTabNet we have computed missing bounding boxes for $48\%$ of the simple and $69\%$ of the complex tables. Regarding FinTabNet, $68\%$ of the simple and $98\%$ of the complex tables require the generation of bounding boxes.  

Figure 7 illustrates the distribution of the tables across different dimensions per dataset.  

# 1.2. Synthetic datasets  

Aiming to train and evaluate our models in a broader spectrum of table data we have synthesized four types of datasets. Each one contains tables with different appearances in regard to their size, structure, style and content. Every synthetic dataset contains $150\mathrm{k}$ examples, summing up to $600\mathrm{k}$ synthetic examples. All datasets are divided into Train, Test and Val splits $(80\%,10\%,10\%)$ .  

The process of generating a synthetic dataset can be decomposed into the following steps:  

1. Prepare styling and content templates: The styling templates have been manually designed and organized into groups of scope specific appearances (e.g. financial data, marketing data, etc.) Additionally, we have prepared curated collections of content templates by extracting the most frequently used terms out of non-synthetic datasets (e.g. PubTabNet, FinTabNet, etc.).  

2. Generate table structures: The structure of each synthetic dataset assumes a horizontal table header which potentially spans over multiple rows and a table body that may contain a combination of row spans and column spans. However, spans are not allowed to cross the header - body boundary. The table structure is described by the parameters: Total number of table rows and columns, number of header rows, type of spans (header only spans, row only spans, column only spans, both row and column spans), maximum span size and the ratio of the table area covered by spans.  

3. Generate content: Based on the dataset theme, a set of suitable content templates is chosen first. Then, this content can be combined with purely random text to produce the synthetic content.  

4. Apply styling templates: Depending on the domain of the synthetic dataset, a set of styling templates is first manually selected. Then, a style is randomly selected to format the appearance of the synthesized table.  

5. Render the complete tables: The synthetic table is finally rendered by a web browser engine to generate the bounding boxes for each table cell. A batching technique is utilized to optimize the runtime overhead of the rendering process.  

# 2. Prediction post-processing for PDF documents  

Although TableFormer can predict the table structure and the bounding boxes for tables recognized inside PDF documents, this is not enough when a full reconstruction of the original table is required. This happens mainly due the following reasons:  

![](images/749ca7790743fafd0d4b6f5bd6239a59927e63750a60584e8b26afa34045766b.jpg)  
Figure 7: Distribution of the tables across different dimensions per dataset. Simple vs complex tables per dataset and split, strict vs non strict html structures per dataset and table complexity, missing bboxes per dataset and table complexity.  

• TableFormer output does not include the table cell content.   
• There are occasional inaccuracies in the predictions of the bounding boxes.  

However, it is possible to mitigate those limitations by combining the TableFormer predictions with the information already present inside a programmatic PDF document. More specifically, PDF documents can be seen as a sequence of PDF cells where each cell is described by its content and bounding box. If we are able to associate the PDF cells with the predicted table cells, we can directly link the PDF cell content to the table cell structure and use the PDF bounding boxes to correct misalignments in the predicted table cell bounding boxes.  

Here is a step-by-step description of the prediction postprocessing:  

1. Get the minimal grid dimensions - number of rows and   
columns for the predicted table structure. This represents   
the most granular grid for the underlying table structure. 2. Generate pair-wise matches between the bounding   
boxes of the PDF cells and the predicted cells. The Intersec  
tion Over Union (IOU) metric is used to evaluate the quality   
of the matches. 3. Use a carefully selected IOU threshold to designate   
the matches as “good” ones and “bad” ones. 3.a. If all IOU scores in a column are below the thresh  
old, discard all predictions (structure and bounding boxes)   
for that column. 4. Find the best-fitting content alignment for the pre  
dicted cells with good IOU per each column. The alignment   
of the column can be identified by the following formula:  

$$
\begin{array}{l}{{a l i g n m e n t=\mathrm{arg}\operatorname*{min}\{D_{c}\}}}\ {{c}}\ {{D_{c}=m a x\{x_{c}\}-m i n\{x_{c}\}}}\end{array}
$$  

where $c$ is one of $\{$ left, centroid, $\mathrm{right}\}$ and $x_{c}$ is the $\mathbf{X}\cdot\mathbf{\partial}$ - coordinate for the corresponding point.  

5. Use the alignment computed in step 4, to compute the median $x$ -coordinate for all table columns and the median cell size for all table cells. The usage of median during the computations, helps to eliminate outliers caused by occasional column spans which are usually wider than the normal.  

6. Snap all cells with bad IOU to their corresponding median $x$ -coordinates and cell sizes.  

7. Generate a new set of pair-wise matches between the corrected bounding boxes and PDF cells. This time use a modified version of the IOU metric, where the area of the intersection between the predicted and PDF cells is divided by the PDF cell area. In case there are multiple matches for the same PDF cell, the prediction with the higher score is preferred. This covers the cases where the PDF cells are smaller than the area of predicted or corrected prediction cells.  

8. In some rare occasions, we have noticed that TableFormer can confuse a single column as two. When the postprocessing steps are applied, this results with two predicted columns pointing to the same PDF column. In such case we must de-duplicate the columns according to highest total column intersection score.  

9. Pick up the remaining orphan cells. There could be cases, when after applying all the previous post-processing steps, some PDF cells could still remain without any match to predicted cells. However, it is still possible to deduce the correct matching for an orphan PDF cell by mapping its bounding box on the geometry of the grid. This mapping decides if the content of the orphan cell will be appended to an already matched table cell, or a new table cell should be created to match with the orphan.  

9a. Compute the top and bottom boundary of the hori  
zontal band for each grid row (min/max $y$ coordinates per   
row). 9b. Intersect the orphan’s bounding box with the row   
bands, and map the cell to the closest grid row. 9c. Compute the left and right boundary of the vertical   
band for each grid column (min/max $x$ coordinates per col  
umn). 9d. Intersect the orphan’s bounding box with the column   
bands, and map the cell to the closest grid column. 9e. If the table cell under the identified row and column  

is not empty, extend its content with the content of the or  

phan cell.  

9f. Otherwise create a new structural cell and match it wit the orphan cell.  

Aditional images with examples of TableFormer predictions and post-processing can be found below.  

TableFormer predicted structure   


<html><body><table><tr><td>0[0. 0] Variable</td><td>Oddsratio</td><td>1 [1, 0] 95%confidence interval</td><td>2 [2.0] pvalue</td><td>3 [3. 0]</td></tr><tr><td>4 [0. 1] Majorvascular complications</td><td>3.91</td><td>5 [1. 1] 1.67-9.14</td><td>6[2. 1] 0.013</td><td>7[3.1]</td></tr><tr><td>8[0.2] Renalfailure requiredCRRT</td><td>10.98</td><td>9[1.2] 6.21-19.41</td><td>10[2.2] <0.001</td><td>11 3.2]</td></tr><tr><td>120.3 Severebleeding</td><td>13[1.3] 15.86</td><td>3.61-69.63</td><td>14[2.3] <0.001</td><td>153.3]</td></tr><tr><td>16[0.4] Neurologic complications</td><td>17 [1.4] 13.68</td><td>5.38-34.80</td><td>18[2.4] <0.001</td><td>193.4]</td></tr></table></body></html>  

![](images/48daa212ae74410d616023a349b00c94e3641226ddec3446e4e2914504d9983b.jpg)  
Figure 9: Example of a table with big empty distance between cells.  

PDF Cells  

<html><body><table><tr><td>Name</td><td>D00 Sequences</td></tr><tr><td>20</td><td>3p</td></tr><tr><td>KRAS-F 4p司</td><td>5-TGTGTGACATGTTCTAATATAGTCACATTT-3 5</td></tr><tr><td>KRAS-R</td><td>5'-ATCGTCAAGGCACTCTTGCCTAC-3</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>PNAclampprobe</td><td>5'-TACGCCACCAGCTCC-3</td></tr></table></body></html>  

TableFormer predicted structure   


<html><body><table><tr><td colspan="5">ANOVA</td></tr><tr><td></td><td>bsun</td><td>Df</td><td>Value</td><td>Fr (>F)</td></tr><tr><td></td><td>E745.2</td><td></td><td>66.75</td><td>4%644</td></tr><tr><td>apnd</td><td>1491.39</td><td></td><td>56.87</td><td>23762033</td></tr><tr><td>26nd</td><td>2648.33</td><td>28</td><td>34.48</td><td>0074</td></tr><tr><td>Residuals</td><td>266.91</td><td>[]</td><td>明</td><td>明</td></tr></table></body></html>  

PDF Cells   


<html><body><table><tr><td></td><td colspan="3">ANOVA</td><td></td></tr><tr><td></td><td>SumSq</td><td>SDf</td><td>FValue</td><td>Pr (>F)</td></tr><tr><td></td><td>5745.2</td><td></td><td>266.75</td><td>64×10</td></tr><tr><td>ponc</td><td>2191.39</td><td></td><td>50.87</td><td>276×10</td></tr><tr><td>1Kcon</td><td>1648.33</td><td></td><td>61.48</td><td>1.07×10</td></tr><tr><td>Residuals</td><td>256.91</td><td>5</td><td>明</td><td>国</td></tr></table></body></html>  

![](images/bcbfd0a0a0717586efc8a4e2c01fb17475aa0872afad392783b4dd9d303bfd9b.jpg)  
Figure 10: Example of a complex table with empty cells.  

Figure 8: Example of a table with multi-line header.   
Post-processed bounding boxes   


<html><body><table><tr><td colspan="5">ANOVA</td></tr><tr><td></td><td>bsun</td><td>DI</td><td></td><td>Fr(>F)</td></tr><tr><td>国</td><td>E745.2</td><td></td><td>E56.75</td><td></td></tr><tr><td></td><td>1191.39</td><td>西回西</td><td>10.87</td><td>76 20 02 13</td></tr><tr><td>2onc</td><td>E48</td><td></td><td></td><td>237023</td></tr><tr><td>esidmals</td><td>246.91</td><td></td><td>明</td><td></td></tr></table></body></html>  

TableFormer predicted bounding boxes   
TableFormerpredicted structure   


<html><body><table><tr><td>00, ANOVA</td><td colspan="3"></td><td>1[1, 2[4,0]</td></tr><tr><td>3p.</td><td>4[1. SumSq</td><td>5（2 Df</td><td>6.1 FValue</td><td>7[4.] Pr(>F)</td></tr><tr><td>8p.2] P</td><td>9[1.2 5745.2</td><td>10[2.2 1</td><td>11p.21 266.75</td><td>12(4,2] 4.64x10-</td></tr><tr><td>13 conc</td><td>14[. 2191.39</td><td>152.3 2</td><td>16p. 50.87</td><td>17 [4.3] 2.76x10-</td></tr><tr><td>18p,4 Pxconc</td><td>19 [1,4] 2648.33</td><td>20[2.4] 2</td><td>61.48</td><td>22 [4, 4] 1.07x10-</td></tr><tr><td>230, Residuals</td><td>24[1.] 236.91</td><td>25[2.号 11</td><td>26,</td><td>274,</td></tr></table></body></html>  

PDF Cells   


<html><body><table><tr><td></td><td>omM ng/ml/islet)</td><td>6. TEM ag/ml/islet)</td><td>Fold-increase (high GLC wGLC</td></tr><tr><td></td><td></td><td>196 15</td><td></td></tr><tr><td>gliptin treatec</td><td>210e</td><td></td><td></td></tr><tr><td>sbontro</td><td></td><td>B46</td><td>1333</td></tr><tr><td>A iptin treated</td><td>43</td><td></td><td></td></tr></table></body></html>  

TableFormer predicted bounding boxes   


<html><body><table><tr><td></td><td>3mM (ng/ml/islet)</td><td>16.7mM (ng/ml/islet)</td><td>Fold-increase (high GLC/ iewGLC</td></tr><tr><td>Bocontrol</td><td>17+007</td><td>196±0.27</td><td>453</td></tr><tr><td>Bovildagliptin treated</td><td>048±0.06</td><td>4034±0.32</td><td></td></tr><tr><td>KaKA'control</td><td>4±0.041</td><td>F206±0.07</td><td>Bo13</td></tr><tr><td>KAvildagliptin treated</td><td>50±0.07</td><td>2.81±0.301</td><td>67121</td></tr></table></body></html>  

Post-processed bounding boxes   


<html><body><table><tr><td></td><td>mM</td><td>6./1UM</td><td>ghGLC</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>421242</td><td></td></tr><tr><td>2A3bontro</td><td></td><td></td><td></td></tr><tr><td>A</td><td></td><td></td><td></td></tr></table></body></html>  

TableFormer predicted structure   


<html><body><table><tr><td></td><td></td><td>22.</td><td>Fold-increase</td></tr><tr><td>4p</td><td>5 11.11 3mM</td><td>62 16.7mM</td><td>7%11 (highGLC/</td></tr><tr><td>p2</td><td>9.2 (ng/ml/islet)</td><td>10 (ng/ml/islet)</td><td>11p low GLC)</td></tr><tr><td>12p.3] B6control</td><td>131.3 0.47±0.07</td><td>14[2. 2.96±0.27</td><td>15 6.63</td></tr><tr><td>16.4 B6vildagliptin treated</td><td>1701.4 0.48±0.06</td><td>18 [2.4] 4.34±0.32</td><td>19 (3,4] 9.43.</td></tr><tr><td>200.9 KKAycontrol</td><td>21 [1. 0.34±0.04</td><td>1.06±0.07.</td><td>23执 3.43.</td></tr><tr><td>24p. KKAyvildagliptin treated</td><td>2511. 0.30±0.07</td><td>25司 1.81±0.30</td><td>27p 6.42t</td></tr></table></body></html>  

PDF Cells   


<html><body><table><tr><td>reatment</td><td>ank number</td><td>CO,(atm)</td><td></td><td>fotal alkalinity foholkgoaz</td><td>salinity (ppt)</td><td>emperature (°C</td></tr><tr><td>Contro</td><td></td><td>397 122.5</td><td>90009</td><td>214527</td><td>35.6 2U.07</td><td>28.6 500</td></tr><tr><td>Cbntro</td><td></td><td>384 505.8</td><td>87180.006</td><td>29457</td><td>55.6 0.07</td><td>2B.4 00.04</td></tr><tr><td>Mledium</td><td></td><td>01416.6</td><td>8:00 0.009</td><td>209531</td><td>650 k0.07</td><td>2B.7 0.05</td></tr><tr><td>Medium</td><td></td><td>60876.5</td><td>8000.009</td><td>209591</td><td>35.90.07</td><td>2B.6 0.05</td></tr><tr><td>High</td><td></td><td>87614.6</td><td>B610.006</td><td>20793</td><td>36.0 0.03</td><td>26.7.03</td></tr><tr><td>Hgh</td><td></td><td>88% 14.4</td><td>787±.006</td><td>207915</td><td>36.0 113.03</td><td>26.7 号.04</td></tr></table></body></html>  

TableFormerpredicted bounding boxes   


<html><body><table><tr><td>Treatment</td><td>Tank number</td><td>PCO, (μatm)</td><td>PH</td><td>Total alkalinity (μmol kg-)</td><td>Salinity (ppt)</td><td>Temperature (°C)</td></tr><tr><td>Control</td><td></td><td>397 ± 6.5</td><td>846 ± 0.006</td><td>2145 ± 4.7</td><td>35.6 ± 0.07</td><td>28.6 ± 0.05</td></tr><tr><td>Gontrol</td><td></td><td>384±6.8</td><td>818±0.006</td><td>0145±4.7</td><td>85.6±0.07</td><td>26.4±0.04</td></tr><tr><td>Medlum</td><td></td><td>614 ± 16.6</td><td>8:00±0:009</td><td>2095 ± 5.1</td><td>235.9±0.07</td><td>28.7±0.05</td></tr><tr><td>Medium</td><td></td><td>08 ± 16.5</td><td>8:00±0.009</td><td>E2095 ±5.1</td><td>35.9±0.07</td><td>28.6 ±0.05</td></tr><tr><td>Hiqh</td><td></td><td>876 ± 14.6</td><td>两86±0.006</td><td>F2079 ± 5.3</td><td>436.0 ±0.03</td><td>28.7 ± 0.03</td></tr><tr><td>LHiqh</td><td>2</td><td>661 ± 14.4]</td><td>.87±0.006</td><td>2079±5.3</td><td>36.0±0.03</td><td>28.7±0.04</td></tr></table></body></html>  

![](images/c1224e60f3733cb63291098c308294979118eb36cc2d3dac06ded6c462fe3fdc.jpg)  
Figure 11: Simple table with different style and empty cells.  

Post-processed bounding boxes   


<html><body><table><tr><td>reatment</td><td>ank numbe</td><td>CO,(Eatm</td><td></td><td>otal alkalinity lctnolkgfy</td><td>linity(ppt</td><td>mperature(°C</td></tr><tr><td>bntro</td><td></td><td>9716.</td><td>160.006</td><td>454.</td><td>1 5.60.07</td><td>18.60.05</td></tr><tr><td>14 ontro</td><td></td><td>384E6.</td><td>18E0.006</td><td>45年</td><td>560.07</td><td></td></tr><tr><td>edium</td><td></td><td>4816.6</td><td>500000</td><td>20955.1</td><td></td><td>*6.70.05</td></tr><tr><td>tledium</td><td></td><td>10816.5</td><td>000.009</td><td>95</td><td>.90.07</td><td>.60.05</td></tr><tr><td></td><td></td><td>76E14.6</td><td>860.004</td><td>207915.3</td><td>95.00.03</td><td>28.70.03</td></tr><tr><td></td><td>中</td><td>101E44.4</td><td>370.006</td><td>157919</td><td>96.018.03</td><td>B.71304</td></tr></table></body></html>  

PDF Cells   


<html><body><table><tr><td>ariable</td><td>Sensitivity (%)</td><td>Specificity (%)</td><td>Gutoff</td></tr><tr><td>DotalBilirubin</td><td>60</td><td>5</td><td>.3 mg/dl</td></tr><tr><td>DirectBilirubin</td><td>60]</td><td></td><td>085</td></tr><tr><td>ReacuveProtein</td><td></td><td>85</td><td>98</td></tr></table></body></html>  

TableFormer predicted bounding boxes   


<html><body><table><tr><td>Wariable</td><td>Sensitivity (%</td><td>Specificity (%)</td><td>Cutoff</td></tr><tr><td>TotalBilirubin</td><td>60</td><td>95</td><td>1,3mg/dl</td></tr><tr><td>DirectBiirubin</td><td>60</td><td>95</td><td>0,85</td></tr><tr><td>C-ReactiveProtein</td><td>47</td><td>85</td><td>08</td></tr></table></body></html>  

Post-processed boundingboxes   


<html><body><table><tr><td>ariable</td><td>sensitivity(%</td><td>peciticity（%</td><td></td></tr><tr><td>otalBilirubin</td><td></td><td></td><td>.3mq/dl</td></tr><tr><td>DirectBilirubin</td><td></td><td></td><td>0785</td></tr><tr><td>12 ReactveProtein</td><td>国</td><td></td><td></td></tr></table></body></html>  

Figure 13: Table predictions example on colorful table.   


<html><body><table><tr><td>Cortical Laver</td><td>Grade4</td><td>Grades</td><td>Grade2</td><td>Gradel</td></tr><tr><td>srolecuar</td><td>Cnitomly thick and oellular</td><td>Canable thinning,nonmal aellulanty</td><td>sonable thinning and aduced oeluianty</td><td></td></tr><tr><td>Harcnje</td><td>well populated with sologicallvintact peramsdal neurons</td><td>hotatedneuronal loss or ssinophilic degemerabon ecTosIs</td><td>puesdeb aepo saattered loss of neurons</td><td>karge gaps and seuronal necrosis pasearoutaisnonotdsuce</td></tr><tr><td>Gaanue</td><td>Zniomly Uck and densely cellular</td><td>aweguiaruhinning but densely cellular</td><td>aeegular thunning with TeO ut sononpa12Sapce density</td><td>aregular thinning and ouspicuousreducuonsin oell density</td></tr></table></body></html>  

TableFormer predicted structure   


<html><body><table><tr><td>0 (0.9]</td><td>1[1.0] Sensitivity(%)</td><td>Specificity(%)</td><td>Cutoff</td><td>3p.</td></tr><tr><td>4 [0,] Total Bilirubin</td><td>5[1,] 60</td><td>6.] 95</td><td>1,3 mg/dl</td><td>7 p]</td></tr><tr><td>8.21 DirectBilirubin</td><td>9[,2] 60</td><td>102.2 95</td><td>0,85</td><td>11.2]</td></tr><tr><td>12,3] C-Reactive Protein</td><td>13日3 47</td><td>14[.3] 85</td><td>86</td><td>15,3]</td></tr></table></body></html>  

TableFormer predicted structure   
TableFormer predicted bounding boxes   


<html><body><table><tr><td>Cortical Layer</td><td>Grade 4</td><td>Grade3</td><td>Grade2</td><td>Grade1</td></tr><tr><td>sMolecular</td><td>Uniformly thick and cellular</td><td>Wanable thinning, normal cellulanity</td><td>sVanable thinning and reduced cellularity</td><td>Uniformly thin</td></tr><tr><td>Purkinje</td><td>Mstologically intact Well populated with pyramidal neurons</td><td>sotatedneuronal loss or eosinophilic degeneration (necrosis)</td><td>Moderate gaps and scattered loss of neumns</td><td>pue sdef afirefr conspicuously increased neuronal necrosis</td></tr><tr><td>iSranule</td><td>Bniformly thick and densely cellular</td><td>Rmegular thinning but densely cellular</td><td>Imegular-thinning-with modest reductions in cel density</td><td>regular thinning and conspicuous reductions in cell density</td></tr></table></body></html>  

PDF Cells   
Post-processed bounding boxes   
TableFormer predicted structure   


<html><body><table><tr><td>borucalLave</td><td>srades</td><td>Grades</td><td>bradez</td><td>arade</td></tr><tr><td></td><td>ellular</td><td></td><td></td><td></td></tr><tr><td>amle</td><td>Lologicallvntac</td><td>TEODEFOSSOE mophulicdegeneraton</td><td>Sooerale gsps and nienedlossotneuron</td><td>uesebab</td></tr><tr><td>anul</td><td>nselvoelmar</td><td></td><td>modest reductions in cell ensity</td><td>sanspicuousreduchonsin</td></tr></table></body></html>  

![](images/888ce40993d83a9fef74b6a94993d1a4ca74cc721795b3f850d25940fcebe582.jpg)  
Figure 12: Simple table predictions and post processing.   
Figure 14: Example with multi-line text.  

![](images/5fa2b77467d171259b5a96c03e4189ff247e48a9cfa990c5b8d9171dd3875dd4.jpg)  

![](images/b1f71391e57f3a4dd5ec5b69b43910cf246b2404b320943459fa120727104aeb.jpg)  

![](images/6c0bb120f526fbe51338e363b36f4b2fc1914ae3c9f2ec318f49c9330b3a2764.jpg)  

PDF Cells   
TableFormer predicted bounding boxes   
Post-processed bounding boxes   
TableFormer predicted structure   
TableFormer predicted bounding boxes   
Post-processed bounding boxes   
TableFormerpredicted structure   


<html><body><table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Gain</td><td>2851V/V (69.09dB)</td></tr><tr><td>LowCut-OffFrequency</td><td>285Hz</td></tr><tr><td>HighCut-OffFrequency</td><td>6580Hz</td></tr><tr><td>Pnput-ReferredNoise</td><td>.1 eeV (rms)</td></tr><tr><td>@MRR</td><td>p10dB@1KHz</td></tr><tr><td>MumberofAnalogChannels</td><td></td></tr><tr><td>PowerConsumption</td><td>1mA@3.0V(3mW)</td></tr><tr><td>Precision</td><td>gelectable,12or8bits</td></tr></table></body></html>  

![](images/c957aba0462a6addfbb2d2bb4b7a07467eca44b387eee57214b5e60ca92bb334.jpg)  

![](images/c8dbae9046c4426dd3ca85faf2343f981ff5815465847ef3d02b51e3caa5d3eb.jpg)  

![](images/828a9aca4a893b415afd0630bf183b7d9f9bea30f3e389d3285abf577593188f.jpg)  
Figure 15: Example with triangular table.  

![](images/3b2874a72d6047e798f4f35acaa6b66a000c06a271fe9f635a0309b82d718150.jpg)  
Figure 16: Example of how post-processing helps to restore mis-aligned bounding boxes prediction artifact.  

![](images/c6980dddf4de39161d2d35a05253f04022bf89726ae0a5b3a68b3031cfdc647e.jpg)  

Figure 17: Example of long table. End-to-end example from initial PDF cells to prediction of bounding boxes, post processing and prediction of structure.  