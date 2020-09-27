from tensorflow.io.gfile import GFile

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2

from tensorflow.keras.layers import Concatenate, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras import Model

from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import l1_l2

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

def create_text_model(max_seq_len, bert_ckpt_file, bert_config_file, NUM_CLASS, overwriteLayerAndEmbeddingSize = False, isPreTrained=False, pathToBertModelWeights=None, isTrainable=True ):
    with GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        
        if overwriteLayerAndEmbeddingSize:
            bc.max_position_embeddings = max_seq_len
        
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")
        
    input_ids = Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = Lambda(lambda seq: seq[:, 0, :], name = 'bert_output_layer_768')(bert_output)
    cls_out = Dropout(0.5)(cls_out) 
    output = Dense(NUM_CLASS, activation="softmax")(cls_out) #

    model_bert = Model(inputs=input_ids, outputs=output, name = 'BERT')
    model_bert.build(input_shape=(None, max_seq_len))
    
    if not isPreTrained:
        load_stock_weights(bert, bert_ckpt_file) 
        return model_bert
    else:
        model_bert.load_weights(pathToBertModelWeights)
        if not isTrainable:
            for layer in model_bert.layers:
                layer.trainable = False
        return model_bert, 2
        
    
def create_image_resnet50v2_model(NUM_CLASS, isPreTrained=False, pathToResNet50V2ModelWeights=None):
    resnet50v2 = ResNet50V2(weights='imagenet', include_top=False)
    last_layer = resnet50v2.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(768, activation='relu', name='img_dense_768')(x)
    out = Dense(NUM_CLASS, activation='softmax', name='img_output_layer')(x)
    resnet50v2 = Model(inputs=resnet50v2.input, outputs=out, name="ResNet50V2")
    
    if not isPreTrained:
        return resnet50v2              
    else:
        resnet50v2.load_weights(pathToResNet50V2ModelWeights)
        return resnet50v2, 2
    
    
def create_image_resnet101v2_model(NUM_CLASS, isPreTrained=False, pathToResNet101V2ModelWeights=None):
    resnet101v2 = ResNet101V2(weights='imagenet', include_top=False)
    last_layer = resnet101v2.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(768, activation='relu', name='img_dense_768')(x)
    out = Dense(NUM_CLASS, activation='softmax', name='img_output_layer')(x)
    resnet101v2 = Model(inputs=resnet101v2.input, outputs=out, name="ResNet50V2")
    
    if not isPreTrained:
        return resnet101v2              
    else:
        resnet101v2.load_weights(pathToResNet101V2ModelWeights)
        return resnet101v2, 2
    
def create_image_inceptionv3_model(NUM_CLASS, isPreTrained=False, pathToInceptionV3ModelWeights=None, isTrainable=True):
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
    last_layer = inceptionv3.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(768, activation='relu', name='img_dense_768')(x)
    out = Dense(NUM_CLASS, activation='softmax', name='img_output_layer')(x)
    inceptionv3 = Model(inputs=inceptionv3.input, outputs=out, name="InceptionV3")
    
    if not isPreTrained:
        return inceptionv3              
    else:
        inceptionv3.load_weights(pathToInceptionV3ModelWeights)
        if not isTrainable:
            for layer in inceptionv3.layers:
                layer.trainable = False
        return inceptionv3, 2
    

def create_model_meta(NUM_CLASS, shape, isPreTrained=False, pathToMetaModelWeights=None, isTrainable=True):
    initializer = GlorotNormal()
    inputs = Input(shape=(shape))   
    x = Dense(60, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), kernel_initializer=initializer)(inputs)
    x = Dense(30, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), kernel_initializer=initializer)(x)
    x = Dense(6, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), kernel_initializer=initializer, name='final_output')(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs, output)
        
    if not isPreTrained:
        return model              
    else:
        model.load_weights(pathToMetaModelWeights)
        if not isTrainable:
            for layer in model.layers:
                layer.trainable = False
        return model, 1
    
    
    
def buildConcatModelTitleImage(model_image, model_bert, NUM_CLASS, isPreTrained=False, pathToModelWeights=None, isTrainable=True):
    model_image_output = model_image.get_layer('img_dense_768').output
    model_bert_output = model_bert.get_layer('bert_output_layer_768').output
    
    # Build new models, without softmax
    model_img = Model(model_image.inputs, model_image_output)    
    model_bert = Model(model_bert.inputs, model_bert_output)
    
    concatenate = Concatenate()([model_img.output, model_bert.output]) # Fusion
    x = Dense(512, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model = Model([model_image.input, model_bert.input], output)

    if not isPreTrained:
        return model             
    else:
        model.load_weights(pathToModelWeights)
        if not isTrainable:
            for layer in model.layers:
                layer.trainable = False
        return model, 3


def buildConcatModelCommentsImage(model_image, model_bert, NUM_CLASS):
    model_image_output = model_image.get_layer('img_dense_768').output
    model_bert_output = model_bert.get_layer('bert_output_layer_768').output
    
    # Build new models, without softmax
    model_img = Model(model_image.inputs, model_image_output)    
    model_bert = Model(model_bert.inputs, model_bert_output)
    
    concatenate = Concatenate()([model_img.output, model_bert.output]) # Fusion
    x = Dense(512, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model_concat = Model([model_image.input, model_bert.input], output)
    return model_concat

def buildConcatModelTitleComments(model_bert_title, model_bert_comments, NUM_CLASS):
    model_bert_title_output = model_bert_title.get_layer('bert_output_layer_768_title').output
    model_bert_comments_output = model_bert_comments.get_layer('bert_output_layer_768').output
    
    # Build new models, without softmax
    model_bert_title = Model(model_bert_title.inputs, model_bert_title_output)    
    model_bert_comments = Model(model_bert_comments.inputs, model_bert_comments_output)
    
    concatenate = Concatenate()([model_bert_title.output, model_bert_comments.output]) # Fusion
    x = Dense(512, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model_concat = Model([model_bert_title.input, model_bert_comments.input], output)
    return model_concat

def buildConcatModelTitleMeta(model_bert_title, model_meta, NUM_CLASS):
    model_bert_title_output = model_bert_title.get_layer('bert_output_layer_768').output
    model_meta_output = model_meta.get_layer('final_output').output
    
    # Build new models, without softmax
    model_bert_title = Model(model_bert_title.inputs, model_bert_title_output)    
    model_meta = Model(model_meta.inputs, model_meta_output)
    
    x = Dense(128, activation = 'relu')(model_bert_title.output)
    x = Dense (32, activation = 'relu')(x)
    x = Dense (6, activation = 'relu')(x)
    concatenate = Concatenate()([model_meta.output, x]) # Fusion
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model_concat = Model([model_bert_title.input, model_meta.input], output)
    return model_concat


def buildConcatModelMetaVisual(model_meta, model_image, NUM_CLASS):
    model_meta_output = model_meta.get_layer('final_output').output
    model_image_output = model_image.get_layer('img_dense_768').output
    
    # Build new models, without softmax
    model_image = Model(model_image.inputs, model_image_output)    
    model_meta = Model(model_meta.inputs, model_meta_output)
    
    x = Dense(128, activation = 'relu')(model_image.output)
    x = Dense (32, activation = 'relu')(x)
    x = Dense (6, activation = 'relu')(x)
    concatenate = Concatenate()([model_meta.output, x]) # Fusion
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model_concat = Model([model_meta.input, model_image.input], output)
    return model_concat

def buildConcatModelImageTitleMeta(model_bert_title, model_image, model_meta, NUM_CLASS):
    model_bert_title_output = model_bert_title.get_layer('bert_output_layer_768_title').output
    model_meta_output = model_meta.get_layer('final_output').output
    model_image_output = model_image.get_layer('img_dense_768').output
    
    # Build new models, without softmax
    model_bert_title = Model(model_bert_title.inputs, model_bert_title_output)    
    model_image = Model(model_image.inputs, model_image_output)    
    model_meta = Model(model_meta.inputs, model_meta_output)
    
    # Build multimodal model
    
    concatenate = Concatenate()([model_bert_title.output, model_meta.output, model_image_output]) # Fusion
    x = Dense(1024, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    x = Dense(6, activation = 'relu')(x)
    concatenateMeta = Concatenate()([model_meta.output, x]) # Fusion
    x = Dropout(0.4)(concatenateMeta)
   
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model_concat = Model([model_image.input, model_bert_title.input, model_meta.input], output)
    return model_concat

def buildConcatModelTitleCommentsVisual(model_bert_title, model_bert_comments, model_image, NUM_CLASS, isPreTrained=False, pathToModelWeights=None, isTrainable=True):
    model_bert_title_output = model_bert_title.get_layer('bert_output_layer_768_title').output
    model_bert_comments_output = model_bert_comments.get_layer('bert_output_layer_768').output
    model_image_output = model_image.get_layer('img_dense_768').output
    
    # Build new models, without softmax
    model_bert_title = Model(model_bert_title.inputs, model_bert_title_output)    
    model_bert_comments = Model(model_bert_comments.inputs, model_bert_comments_output)
    model_image = Model(model_image.inputs, model_image_output)    
    
    # Build multimodal model
    
    concatenate = Concatenate()([model_bert_title.output, model_bert_comments.output, model_image_output]) # Fusion
    x = Dense(1024, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    x = Dense(256, activation = 'relu')(x) 
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model = Model([model_image.input, model_bert_title.input, model_bert_comments.input], output)
    
    if not isPreTrained:
        return model             
    else:
        model.load_weights(pathToModelWeights)
        if not isTrainable:
            for layer in model.layers:
                layer.trainable = False
        return model, 4


def buildConcatModelTitleCommentsMeta(model_bert_title, model_bert_comments, model_meta, NUM_CLASS):
    model_bert_title_output = model_bert_title.get_layer('bert_output_layer_768_title').output
    model_bert_comments_output = model_bert_comments.get_layer('bert_output_layer_768').output
    model_meta_output = model_meta.get_layer('final_output').output
    
    # Build new models, without softmax
    model_bert_title = Model(model_bert_title.inputs, model_bert_title_output)    
    model_bert_comments = Model(model_bert_comments.inputs, model_bert_comments_output)    
    model_meta = Model(model_meta.inputs, model_meta_output)
    
    # Build multimodal model
    
    concatenate = Concatenate()([model_bert_title.output, model_bert_comments.output]) # Fusion
    x = Dense(1024, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    x = Dense(6, activation = 'relu')(x)
    concatenateMeta = Concatenate()([model_meta.output, x]) # Fusion
    x = Dropout(0.4)(concatenateMeta)
   
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model_concat = Model([model_bert_title.input, model_bert_comments.input, model_meta.input], output)
    return model_concat

def buildConcatModelTitleCommentsMetaVisual(model_bert_title, model_bert_comments, model_image, model_meta, NUM_CLASS, isPreTrained=False, pathToModelWeights=None, isTrainable=True):
    model_bert_title_output = model_bert_title.get_layer('bert_output_layer_768_title').output
    model_bert_comments_output = model_bert_comments.get_layer('bert_output_layer_768').output
    model_meta_output = model_meta.get_layer('final_output').output
    model_image_output = model_image.get_layer('img_dense_768').output
    
    # Build new models, without softmax
    model_bert_title = Model(model_bert_title.inputs, model_bert_title_output)    
    model_bert_comments = Model(model_bert_comments.inputs, model_bert_comments_output)
    model_image = Model(model_image.inputs, model_image_output)    
    model_meta = Model(model_meta.inputs, model_meta_output)
    
    # Build multimodal model
    
    concatenate = Concatenate()([model_bert_title.output, model_bert_comments.output, model_image_output]) # Fusion
    x = Dense(1024, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    x = Dense(6, activation = 'relu')(x)
    concatenateMeta = Concatenate()([model_meta.output, x]) # Fusion
    x = Dropout(0.4)(concatenateMeta)
   
    output = Dense(NUM_CLASS, activation='softmax', name='final_multimodal_output')(x)
    model = Model([model_image.input, model_bert_title.input, model_bert_comments.input, model_meta.input], output)
    
    if not isPreTrained:
        return model             
    else:
        model.load_weights(pathToModelWeights)
        if not isTrainable:
            for layer in model.layers:
                layer.trainable = False
        return model, 7
