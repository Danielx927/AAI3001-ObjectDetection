## Data Collection & Annotation

The dataset consists of **1,295 fruit images** across **10 fruit classes**. Data was collected through a combination of self-annotation and external sources to ensure class balance and diversity.

---

### 1. Self-Annotated Images (995 images)

A total of **995 images** were annotated manually using **LabelImg**, of which contains
**around 100 images per class** that were filtered from our previously collected dataset used in the first half of the project on fruit classification. 

---

### 2. External Kaggle Sources (300 images)

To increase variability and representation, **300 additional images** were sourced from **multiple Kaggle datasets**.  
Only images that matched the 10 fruit classes were included.

All imported images were:
- Renamed and reorganized into the project structure
- Re-labelled to ensure consistent class naming
- Checked for duplicates before inclusion
- Converted any non-yolo format labels into yolo format

#### Kaggle Sources Used

Below are the Kaggle datasets referenced (insert actual links in the placeholders):

1. **Dataset Source 1**  
   *Fruit Images for Object Detection:* <https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection>  
   *Notes:* 300 images of apple, banana and orange were taken from this dataset

2. **Dataset Source 2**  
   *Kaggle Link:* <INSERT_LINK_HERE>  
   *Notes:* Mention which classes were extracted.

3. **Dataset Source 3**  
   *Kaggle Link:* <INSERT_LINK_HERE>  
   *Notes:* Optional — only include if used.

*(Add or remove entries depending on how many sources you used.)*

---

This combined dataset provides a well-structured, consistently labelled, and diverse foundation for model training and evaluation.
