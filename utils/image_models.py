
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Lambda, Input, Maximum
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.io.gfile import GFile
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

def build_resnet50(NUM_CLASS):
    resnet50 = ResNet50(weights='imagenet', include_top=False)
    #model.summary()
    last_layer = resnet50.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    # a softmax layer for 2 classes
    out = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    resnet50 = Model(inputs=resnet50.input, outputs=out)
    plot_model(resnet50, to_file='multiple_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
    return resnet50, 7

def build_resNet101V2(NUM_CLASS):
    ResNet101v2 = ResNet101V2(weights='imagenet', include_top=False)
    #model.summary()
    last_layer = ResNet101v2.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    # a softmax layer for 2 classes
    out = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    ResNet101v2 = Model(inputs=ResNet101v2.input, outputs=out)
    plot_model(ResNet101v2, to_file='multiple_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
    return ResNet101v2, 7


def build_resnet50V2(NUM_CLASS):
    resnet50v2 = ResNet50V2(weights='imagenet', include_top=False)
    #model.summary()
    last_layer = resnet50v2.output
    x = GlobalAveragePooling2D()(last_layer)
    # a softmax layer for 2 classes
    x = Dense(768, activation='relu', name='img_dense_768')(x)
    out = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    resnet50v2 = Model(inputs=resnet50v2.input, outputs=out)
    plot_model(resnet50v2, to_file='resnet50v2_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
    return resnet50v2, 7


def build_resnet101(NUM_CLASS):
    resnet101 = ResNet101(weights='imagenet', include_top=False)
    #model.summary()
    last_layer = resnet101.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    # a softmax layer for 2 classes
    out = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    resnet101 = Model(inputs=resnet101.input, outputs=out)
    plot_model(resnet101, to_file='multiple_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
    return resnet101, 7

def build_inceptionV3(NUM_CLASS):
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
    #model.summary()
    last_layer = inceptionv3.output
    # a softmax layer for 2 classes
    out = Dense(NUM_CLASS, activation='softmax',name='output_layer')(last_layer)
    inceptionv3 = Model(inputs=inceptionv3.input, outputs=out)
    plot_model(inceptionv3, to_file='multiple_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
    return inceptionv3, 1


def build_VGG16(NUM_CLASS):
    vgg16 = VGG16(weights='imagenet', include_top=False)
    #model.summary()
    last_layer = vgg16.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    # a softmax layer for 2 classes
    out = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    vgg16 = Model(inputs=vgg16.input, outputs=out)
    plot_model(vgg16, to_file='multiple_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
    return vgg16, 7


def merge_resnet_inceptionv3():
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
    resnet50 = ResNet50(weights='imagenet', include_top=False)   
    res_out = resnet50.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    inc_out = inceptionv3.output
    inc_out = GlobalAveragePooling2D()(inc_out)
    inc_out = Dropout(0.5)(inc_out)
    inc_out = Dense(2048)(inc_out)
    merge = Concatenate()([res_out,inc_out])
    x = Dense(2048)(merge)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    model = Model(inputs=[resnet50.input,inceptionv3.input], outputs=output)
    # plot graph
    plot_model(model, to_file='multiple_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
#     model.summary()

    return model

def merge_resnet_inceptionv3_without_model_head():
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
    resnet50 = ResNet50(weights='imagenet', include_top=False)   
    res_out = resnet50.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    inc_out = inceptionv3.output
    inc_out = GlobalAveragePooling2D()(inc_out)
    inc_out = Dropout(0.5)(inc_out)
    inc_out = Dense(2048)(inc_out)
    merge = Concatenate()([res_out,inc_out])
    
    return resnet50.input, inceptionv3.input, merge


def merge_resnet_inceptionv3flat(NUM_CLASS):
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
    resnet50 = ResNet50(weights='imagenet', include_top=False)   
    res_out = resnet50.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    inc_out = inceptionv3.output
    inc_out = GlobalAveragePooling2D()(inc_out)
    inc_out = Dropout(0.5)(inc_out)
    inc_out = Dense(2048)(inc_out)
    i_flat = Flatten()(inc_out)
    r_flat = Flatten()(res_out)
    merge = Concatenate()([r_flat,i_flat])
    x = Dense(2048)(merge)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    model = Model(inputs=[resnet50.input,inceptionv3.input], outputs=output)
    # plot graph
    plot_model(model, to_file='multiple_inputs.png', show_shapes=True, dpi=600, expand_nested=False)
#     model.summary()

    return model

def createMultiModelMaximum(max_seq_len, bert_ckpt_file, bert_config_file, NUM_CLASS):
    with GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert_layer = BertModelLayer.from_params(bert_params, name="bert")
        
    bert_in = Input(shape=(max_seq_len, ), dtype='int32', name="input_ids_bert")
    bert_inter = bert_layer(bert_in)
    cls_out = Lambda(lambda seq: seq[:, 0, :])(bert_inter)
    cls_out = Dropout(0.5)(cls_out)
    bert_out = Dense(units=768, activation="tanh")(cls_out) # 768 before
    load_stock_weights(bert_layer, bert_ckpt_file)
    
    
    # image models:
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
    resnet50 = ResNet50(weights='imagenet', include_top=False)   
    res_out = resnet50.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(768)(res_out)
    inc_out = inceptionv3.output
    inc_out = GlobalAveragePooling2D()(inc_out)
    inc_out = Dropout(0.5)(inc_out)
    inc_out = Dense(2048)(inc_out)
    inc_out = Dropout(0.5)(inc_out)
    inc_out = Dense(768)(inc_out)
#     merge = Concatenate()([res_out, inc_out, bert_out])
    merge = Maximum()([res_out, inc_out, bert_out])
    
    # restliche Layer
    x = Dense(2048)(merge)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    model = Model(inputs=[resnet50.input,inceptionv3.input, bert_in], outputs=output)
    plot_model(model, to_file='multiple_inputs_text.png', show_shapes=True, dpi=600, expand_nested=False)
    
    return model, 17

def createTextResNet50v2Maximum(max_seq_len, bert_ckpt_file, bert_config_file, NUM_CLASS):
    with GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert_layer = BertModelLayer.from_params(bert_params, name="bert")
        
    bert_in = Input(shape=(max_seq_len, ), dtype='int32', name="input_ids_bert")
    bert_inter = bert_layer(bert_in)
    cls_out = Lambda(lambda seq: seq[:, 0, :])(bert_inter)
    cls_out = Dropout(0.5)(cls_out)
    bert_out = Dense(units=768, activation="tanh")(cls_out) # 768 before
    load_stock_weights(bert_layer, bert_ckpt_file)
    
    
    # image models:
    resNet50v2 = ResNet50V2(weights='imagenet', include_top=False)   
    res_out = resNet50v2.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(768)(res_out)
    merge = Maximum()([res_out, bert_out])    
    # restliche Layer
    x = Dense(2048)(merge)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    model = Model(inputs=[resNet50v2.input, bert_in], outputs=output)
    plot_model(model, to_file='multiple_inputs_text.png', show_shapes=True, dpi=600, expand_nested=False)
    
    return model, 14


def createTextResNet101v2Maximum(max_seq_len, bert_ckpt_file, bert_config_file, NUM_CLASS):
    with GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert_layer = BertModelLayer.from_params(bert_params, name="bert")
        
    bert_in = Input(shape=(max_seq_len, ), dtype='int32', name="input_ids_bert")
    bert_inter = bert_layer(bert_in)
    cls_out = Lambda(lambda seq: seq[:, 0, :])(bert_inter)
    cls_out = Dropout(0.5)(cls_out)
    bert_out = Dense(units=768, activation="tanh")(cls_out) # 768 before
    load_stock_weights(bert_layer, bert_ckpt_file)
    
    
    # image models:
    ResNet101v2 = ResNet101V2(weights='imagenet', include_top=False)   
    res_out = ResNet101v2.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(768)(res_out)
    merge = Maximum()([res_out, bert_out])    
    # restliche Layer
    x = Dense(2048)(merge)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    model = Model(inputs=[ResNet101v2.input, bert_in], outputs=output)
    plot_model(model, to_file='multiple_inputs_text.png', show_shapes=True, dpi=600, expand_nested=False)
    
    return model, 14

def createTextResNet50Maximum(max_seq_len, bert_ckpt_file, bert_config_file, NUM_CLASS):
    with GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert_layer = BertModelLayer.from_params(bert_params, name="bert")
        
    bert_in = Input(shape=(max_seq_len, ), dtype='int32', name="input_ids_bert")
    bert_inter = bert_layer(bert_in)
    cls_out = Lambda(lambda seq: seq[:, 0, :])(bert_inter)
    cls_out = Dropout(0.5)(cls_out)
    bert_out = Dense(units=768, activation="tanh")(cls_out) # 768 before
    load_stock_weights(bert_layer, bert_ckpt_file)
    
    
    # image models:
    resnet50 = ResNet50(weights='imagenet', include_top=False)   
    res_out = resnet50.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(768)(res_out)
    merge = Maximum()([res_out, bert_out])    
    # restliche Layer
    x = Dense(2048)(merge)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    model = Model(inputs=[resnet50.input, bert_in], outputs=output)
    plot_model(model, to_file='multiple_inputs_text.png', show_shapes=True, dpi=600, expand_nested=False)
    
    return model, 14


def createMultiModelConcat(max_seq_len, bert_ckpt_file, bert_config_file, NUM_CLASS):
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert_layer = BertModelLayer.from_params(bert_params, name="bert")
        
    bert_in = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids_bert")
    bert_inter = bert_layer(bert_in)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_inter)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    bert_out = keras.layers.Dense(units=768, activation="tanh")(cls_out) # 768 before
    load_stock_weights(bert_layer, bert_ckpt_file)
    
    
    # image models:
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
    resnet50 = ResNet50(weights='imagenet', include_top=False)   
    res_out = resnet50.output
    res_out = GlobalAveragePooling2D()(res_out)
    res_out = Dropout(0.5)(res_out)
    res_out = Dense(2048)(res_out)
    inc_out = inceptionv3.output
    inc_out = GlobalAveragePooling2D()(inc_out)
    inc_out = Dropout(0.5)(inc_out)
    inc_out = Dense(2048)(inc_out)
    merge = Concatenate()([res_out, inc_out, bert_out])
    
    # restliche Layer
    x = Dense(2048)(merge)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASS, activation='softmax',name='output_layer')(x)
    model = Model(inputs=[resnet50.input,inceptionv3.input, bert_in], outputs=output)
    plot_model(model, to_file='multiple_inputs_text.png', show_shapes=True, dpi=600, expand_nested=False)
    return model