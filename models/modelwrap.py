import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model

class Models:
    """
    Load specific models predefined within the Keras framework
    """
    def __init__(self):
        self.models_ = dict(
            VGG16 = dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3"
            ),
            VGG19 = dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4"
            ),
            DenseNet121 = dict(
                input_shape=(224, 225, 3),
                module_name="densenet121",
                #TODO ensure this matches paper implementation along with Keras implementation
                last_conv_layer="bn"
            ),
            ResNet50 = dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49"
            ),
            InceptionV3 = dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3"
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7c_ac"
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188"
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260"
            ),
        )

    def get_last_conv_layer(self, model_name: str):
        """Return the weights of the last convolutional layer of a given trained model
        
        Arguments:
            model_name {Model} -- trained model name
        """
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name: str):
        """Get the size of the input layer of the model
        
        Arguments:
            model_name {str} -- trained model name
        """

        return self.models_[model_name]["input_shape"][:2]


    def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
                    weights_path=None, input_shape=None):
        """
        Instantiate our model and specify parameters realted to initial weights
        
        Arguments:
            class_names {str} -- [description]
            use_base_weights {bool} -- [description]
            input_shape {tuple} -- [description]
        
        Keyword Arguments:
            model_name {[type]} -- [description] (default: {"DenseNet121":str})
            use_weights_path {[type]} -- [description] (default: {None})
        """

        # Initialize weights of the model using ImageNet weights as specified in the DenseNet paper
        # else do not initialize the weights and we can randomly normally $N(0, I)$ initialize them
        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                f"keras.applications.{self.models_[model_name]['module_name']}"
            ),
            model_name)
        
        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]
        
        img_input = Input(shape=input_shape)

        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        x = base_model.output
        # Last output layer is a dense connected layer with sigmoid activation according to the CheXNet paper
        predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)

        if weights_path == "":
            weights_path = None
        
        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        
        return model