#%%
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm
import os
import random
import pandas as pd
import re
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
#%%
# ontologies = ['regular', 'semanticOntology', 'aggolomerativeOntologyAverage', 'aggolomerativeOntologyComplete',
#               'aggolomerativeOntologySingle', 'aggolomerativeOntologyWard']
ontologies = ['semanticOntology']

ontologyFile = open('./semanticOntology.json')
ontology = json.load(ontologyFile)


#%%
def get_classes(node):
    classes = []
    mapping = {}
    labels = []
    if 'leaf' in node.keys():
        classes.append(node['leaf'])
    for child in node['children']:
        labels.append(child['name'])
        child_classes = get_classes(child)
        classes.extend(child_classes)
        for c in child_classes:
            mapping[c] = child['name']
    node['classes'] = classes
    node['mapping'] = mapping
    node['labels'] = labels
    return classes

get_classes(ontology)

semanticOntologyFile = open('./semanticOntology.json')
semanticOntology = json.load(semanticOntologyFile)
get_classes(semanticOntology)

def find_ancestors(label, ontology, ancestors):
    for c in ontology["children"]:
        if label in c["classes"]:
            ancestors.add(c["name"])
        if "leaf" in c and label == c["leaf"]:
            ancestors.add(c["leaf"].lower())
        find_ancestors(label, c, ancestors)

ancestors_map = {}

for category in semanticOntology["classes"]:
    ancestors = set()
    find_ancestors(category, semanticOntology, ancestors)
    ancestors_map[category] = ancestors
#%%
train_data_dir = './data/Train'
test_data_dir = './data/Val'

folds = 5

for test_fold in range(folds):
    if os.path.exists(train_data_dir):
        shutil.rmtree(train_data_dir)
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)

    for category in tqdm(get_classes(ontology)):
        os.makedirs(train_data_dir + '/' + category + '/')
        for fold in [x for x in range(folds) if x != test_fold]:
            files = os.listdir('./data/' + str(fold) + '/' + category + '/')
            files = [x for x in files if x != 'desktop.ini']
            for filename in files:
                shutil.copy('./data/' + str(fold) + '/' + category + '/' + filename, train_data_dir + '/' + category + '/')

        os.makedirs(test_data_dir + '/' + category + '/')
        files = os.listdir('./data/' + str(test_fold) + '/' + category + '/')
        files = [x for x in files if x != 'desktop.ini']
        for filename in files:
            shutil.copy('./data/' + str(test_fold) + '/' + category + '/' + filename, test_data_dir + '/' + category + '/')
    #%%
    img_height, img_width = 240, 320
    batch_size = 64

    def preprocess(images, labels):
        return tf.keras.applications.resnet50.preprocess_input(images), labels
    
    print('Loading dataset')
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        seed=0,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=None)
    
    class_names = train_dataset.class_names
    
    print('Preprocessing')
    train_dataset = train_dataset.map(preprocess)
    #%%
    def train_nodes(node):
        print(f'Training {node["name"]}')
        n_classes = len(node['children'])
    
        def filter_classes(image, label):
            return tf.py_function(func=lambda y: y in node['classes'], inp=[tf.gather(class_names, tf.math.argmax(label))], Tout=tf.bool)
    
        def relabel(image, label):
            new_label = tf.py_function(func=lambda y: node['labels'].index(node['mapping'][y.numpy().decode()]), inp=[tf.gather(class_names, tf.math.argmax(label))], Tout=tf.int64)
            new_label = tf.one_hot(new_label, depth=n_classes)
            new_label.set_shape([n_classes])
            return image, new_label
    
    
        train_ds = train_dataset.filter(filter_classes)
        train_ds = train_ds.map(relabel)
        train_ds = train_ds.batch(batch_size)
    
        model = Sequential()
        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=(img_height, img_width, 3),
            pooling='avg',
            weights='imagenet'
        )
        resnet.trainable = False
        model.add(resnet)
        model.add(Dense(512, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    
        model.fit(train_ds, epochs=3)
    
        model.save(model_path + node["name"])
    
        for c in node['children']:
            if 'leaf' not in c.keys():
                train_nodes(c)
    
    
    for o in ontologies:
        print(o, test_fold)
        ontologyFile = open('./' + o + '.json')
        ontology = json.load(ontologyFile)
        model_path = './models/' + str(test_fold) + '/' + o + '/'
        get_classes(ontology)
        train_nodes(ontology)

    test_data_dir = './data/Val'
    img_height, img_width = 240, 320

    categories = get_classes(ontology)

    test_df = pd.DataFrame()
    for category in tqdm(categories):
        images = os.listdir(test_data_dir + '/' + category)
        images = [test_data_dir + '/' + category + '/' + x for x in images if x != 'desktop.ini']
        df = pd.DataFrame({'image': images})
        df['recording'] = df['image'].map(lambda x: re.search(r'.*/([a-zA-X]+\d+)_.*\.jpg', x).group(1))
        df['class'] = category
        test_df = pd.concat([test_df, df], ignore_index=True)

    def preprocess_image(image):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.keras.applications.resnet50.preprocess_input(image)
        image = image.numpy()
        return image

    test_df['image'] = [preprocess_image(x) for x in test_df['image']]

    #%%
    prediction_dict = {}
    for category in categories:
        prediction_dict[category] = 1
    test_df['prediction'] = [prediction_dict.copy() for _ in range(len(test_df))]

    test_ds = tf.data.Dataset.from_generator(lambda: test_df['image'], output_signature=(tf.TensorSpec(shape=(240, 320, 3), dtype=tf.float32)))
    test_ds = test_ds.batch(batch_size)

    def test_nodes(node):
        print(f'Testing {node["name"]}')

        model = tf.keras.models.load_model(model_path + node['name'])
        prediction = model.predict(test_ds)

        def prediction_mapping(prediction):
            classification = {}
            for category, label in node['mapping'].items():
                classification[category] = prediction[node['labels'].index(label)]
            return classification

        predictions = [prediction_mapping(x) for x in prediction]

        def update_predictions(prediction):
            for category in node['classes']:
                prediction['prediction'][category] *= predictions[prediction.name][category]

        test_df.apply(update_predictions, axis=1)
        for c in node['children']:
            if 'leaf' not in c.keys():
                test_nodes(c)

    #%%
    def sum_predictions(predictions):
        final_prediction = {}
        for category in categories:
            final_prediction[category] = 0
            for prediction in predictions:
                final_prediction[category] += prediction[category]
        return final_prediction

    for o in ontologies:
        print(o, test_fold)
        ontologyFile = open('./' + o + '.json')
        ontology = json.load(ontologyFile)
        model_path = './models/' + str(test_fold) + '/' + o + '/'
        get_classes(ontology)

        test_nodes(ontology)
        predictions = test_df.groupby('recording').agg({'class': 'first', 'prediction': sum_predictions})
        predictions['prediction'] = predictions['prediction'].map(lambda x: max(x, key=x.get))

        predictions["pred_ancestors"] = predictions.apply(lambda row: len(ancestors_map[row["prediction"]]), axis=1)
        predictions["true_ancestors"] = predictions.apply(lambda row: len(ancestors_map[row["class"]]), axis=1)
        predictions["intersection"] = predictions.apply(lambda row: len(ancestors_map[row["class"]].intersection(ancestors_map[row["prediction"]])), axis=1)

        hP = predictions["intersection"].sum() / predictions["pred_ancestors"].sum()
        hR = predictions["intersection"].sum() / predictions["true_ancestors"].sum()

        # print("hP", hP)
        # print("hR", hR)
        print("hF1", (2*hP*hR)/(hP+hR))
        
        # print(classification_report(predictions['class'], predictions['prediction']))
        print(accuracy_score(predictions['class'], predictions['prediction']))
        print(confusion_matrix(predictions['class'], predictions['prediction']))

# test_data_dir = './data/Val'
# img_height, img_width = 240, 320

# categories = get_classes(ontology)

# test_df = pd.DataFrame()
# for category in tqdm(categories):
#     images = os.listdir(test_data_dir + '/' + category)
#     images = [test_data_dir + '/' + category + '/' + x for x in images if x != 'desktop.ini']
#     df = pd.DataFrame({'image': images})
#     df['recording'] = df['image'].map(lambda x: re.search(r'.*/([a-zA-X]+\d+)_.*\.jpg', x).group(1))
#     df['class'] = category
#     test_df = pd.concat([test_df, df], ignore_index=True)

# def preprocess_image(image):
#     image = tf.io.read_file(image)
#     image = tf.io.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [img_height, img_width])
#     image = tf.keras.applications.resnet50.preprocess_input(image)
#     image = image.numpy()
#     return image

# test_df['image'] = [preprocess_image(x) for x in test_df['image']]

# #%%
# prediction_dict = {}
# for category in categories:
#     prediction_dict[category] = 1
# test_df['prediction'] = [prediction_dict.copy() for _ in range(len(test_df))]

# test_ds = tf.data.Dataset.from_generator(lambda: test_df['image'], output_signature=(tf.TensorSpec(shape=(240, 320, 3), dtype=tf.float32)))
# test_ds = test_ds.batch(batch_size)

# def test_nodes(node):
#     print(f'Testing {node["name"]}')

#     model = tf.keras.models.load_model(model_path + node['name'])
#     prediction = model.predict(test_ds)

#     def prediction_mapping(prediction):
#         classification = {}
#         for category, label in node['mapping'].items():
#             classification[category] = prediction[node['labels'].index(label)]
#         return classification

#     predictions = [prediction_mapping(x) for x in prediction]

#     def update_predictions(prediction):
#         for category in node['classes']:
#             prediction['prediction'][category] *= predictions[prediction.name][category]

#     test_df.apply(update_predictions, axis=1)
#     for c in node['children']:
#         if 'leaf' not in c.keys():
#             test_nodes(c)

# #%%
# def sum_predictions(predictions):
#     final_prediction = {}
#     for category in categories:
#         final_prediction[category] = 0
#         for prediction in predictions:
#             final_prediction[category] += prediction[category]
#     return final_prediction

# for o in ontologies:
#     print(o, test_fold)
#     ontologyFile = open('./' + o + '.json')
#     ontology = json.load(ontologyFile)
#     model_path = './models/' + str(test_fold) + '/' + o + '/'
#     get_classes(ontology)

#     test_nodes(ontology)
#     predictions = test_df.groupby('recording').agg({'class': 'first', 'prediction': sum_predictions})
#     predictions['prediction'] = predictions['prediction'].map(lambda x: max(x, key=x.get))
#     # print(classification_report(predictions['class'], predictions['prediction']))
#     print(accuracy_score(predictions['class'], predictions['prediction']))
#     print(confusion_matrix(predictions['class'], predictions['prediction']))