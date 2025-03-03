import json
import shutil
import os
import pickle
from callback import MultiClassAUROC, MultiGPUModelCheckpoint
from configparser import ConfigParser
import generator
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from models import modelwrap
import tensorflow as tf
import utility

# hackery 
# this is typically handled using environment variables on host
# configuration; not by scripts
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():

    # Instantiate config parser
    # as long as a configuration file is in the local directory of this training code
    # it will be utilized by the training script

    # TODO : Add a README for the configuration file used to configure this training cycle
    config_file = "./sample_config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # set a bunch of default config
    output_directory = cp["DEFAULT"].get("output_directory")
    image_source_directory = cp["DEFAULT"].get("image_source_directory")
    # TODO tie in base model name for model verioning for SavedModels
    base_model_name = cp["DEFAULT"].get("base_model_name")
    # Class names are passed in as array within the configuration script
    class_names = cp["DEFAULT"].get("class_names").split(",")
    model_version = cp["DEFAULT"].get("model_version")
    tensorboard_log_dir = cp["DEFAULT"].get("tensorboard_log_dir")

    # training configuration
    # See sample_config.ini for explanation of all of the parameters
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    image_dimension = cp["TRAIN"].getint("image_dimension")
    train_steps = cp["TRAIN"].get("train_steps")
    patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
    min_learning_rate = cp["TRAIN"].getfloat("min_learning_rate")
    validation_steps = cp["TRAIN"].get("validation_steps")
    positive_weights_multiply = cp["TRAIN"].getfloat("positive_weights_multiply")
    dataset_csv_dir = cp["TRAIN"].get("dataset_csv_dir")

    if use_trained_model_weights:
        print("<<< Using pretrained model weights! >>>")
        training_stats_file = os.path.join(output_directory, ".training_stats.json")
        if os.path.isfile(training_stats_file):
            training_stats = json.load(open(training_stats_file))
        else: 
            training_stats = {}
    else:
        # start over again
        training_stats = {}
    
    show_model_summary = cp["TRAIN"].getboolean("show_model_summary")
    # end configuration parser

    utility.check_create_output_dir(output_directory)
    utility.create_tensorboard_log_dir(tensorboard_log_dir)
    
    try:

        utility.backup_config_file(output_directory, config_file)

        datasets = ["train", "validation", "test"]
        for dataset in datasets:
            shutil.copy(os.path.join(dataset_csv_dir, f"{dataset}.csv"), output_directory)

        train_counts, train_pos_counts = utility.get_sample_counts(output_directory, "train", class_names)
        validation_counts, _ = utility.get_sample_counts(output_directory, "validation", class_names)

        # compute steps

        # train steps var defined in config ini file
        # if set to standard auto, normalize train_steps
        # wrt batch_size, otherwise take user input
        if train_steps == "auto":
            train_steps = int(train_counts / batch_size)
        else:
            try:
                train_steps = int(train_steps)
            except:
                raise ValueError(f"""
                train_steps : {train_steps} is invalid,
                please use 'auto' or specify an integer.
                """)
            print(f" <<< train_steps : {train_steps} >>>")

        if validation_steps == "auto":
            validation_steps = int(validation_counts / batch_size)
        else:
            try:
                validation_steps = int(validation_steps)
            except:
                raise ValueError(f"""
                validation_steps : {validation_steps} is invalid,
                please use 'auto' or specify an integer.
                """)
        print(f" <<< validation_steps : {validation_steps} >>>")

        # class weights
        class_weights = utility.get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
        )
        print(f"class_weights : {class_weights}")

        print(" <<< Loading Model >>>")
        if use_trained_model_weights:
            if use_best_weights:
                model_weights_file = os.path.join(output_directory, f"best_{output_weights_name}")
            else:
                model_weights_file = os.path.join(output_directory, output_weights_name)
        else:
            model_weights_file = None
        
        model_factory = modelwrap.Models()
        model = model_factory.get_model(
            class_names=class_names,
            use_base_weights=use_base_model_weights,
            weights_path=model_weights_file,
            input_shape=(image_dimension,image_dimension,3)
        )

        if show_model_summary:
            print(model.summary())
        
        print(" <<< Creating Image Generators >>> ")
        train_sequence = generator.AugmentedImageSequence(
            dataset_csv_file=os.path.join(output_directory, "train.csv"),
            class_names=class_names,
            source_image_dir=image_source_directory,
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=utility.augmenter(),
            steps=train_steps,
        )
        
        validation_sequence = generator.AugmentedImageSequence(
            dataset_csv_file=os.path.join(output_directory, "validation.csv"),
            class_names=class_names,
            source_image_dir=image_source_directory,
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=utility.augmenter(),
            steps=validation_steps,
            shuffle_on_epoch_end=False,
        )

        output_weights_path = os.path.join(output_directory, output_weights_name)
        print(f" <<< Set Output Weights Path to : {output_weights_path}")

        # TODO implement multi-gpu support

        model_train = model
        checkpoint = ModelCheckpoint(
            output_weights_path,
            save_weights_only=True,
            save_best_only=True,
            verbose=1
        )

        print(" <<< Compile model and class weights >>>")
        optimizer = Adam(lr=initial_learning_rate)
        model_train.compile(
            optimizer=optimizer, loss="binary_crossentropy"
        )

        auroc = MultiClassAUROC(
            sequence=validation_sequence,
            class_names=class_names,
            weights_path=output_weights_path,
            stats=training_stats,
            workers=generator_workers,
        )

        # serving_checkpoint = ServingCheckpoint(
        #     output_directory=output_directory,
        #     model=model,
        #     model_version=model_version,
        # )

        callbacks =[
            checkpoint,
            TensorBoard(log_dir=os.path.join(tensorboard_log_dir), batch_size=batch_size),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
                            verbose=1, mode="min", min_lr=min_learning_rate),
            auroc
            ]

        print(" <<< Starting Model Training >>> ")
        history = model_train.fit_generator(
            generator=train_sequence,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_sequence,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=generator_workers,
            shuffle=False,
        )

        model_class_weights = tf.convert_to_tensor(model_train.layers[-1].get_weights()[0], tf.float32)
        model_final_conv_layer = utility.get_output_layer(model_train, "bn")
        

        tensor_info_input = tf.saved_model.utils.build_tensor_info(model_train.input)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(model_train.output)
        tensor_info_class_weights = tf.saved_model.utils.build_tensor_info(model_class_weights)
        tensor_info_final_conv_layer = tf.saved_model.utils.build_tensor_info(model_final_conv_layer.output)

        # export model for serving
        export_base_path = output_directory
        export_path = os.path.join(
            tf.compat.as_bytes(export_base_path),
            tf.compat.as_bytes(model_version)
        )

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input},
                outputs={'prediction': tensor_info_output, 'class_weights': tensor_info_class_weights, 'final_conv_layer': tensor_info_final_conv_layer},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        print(f" <<< Exporting Trained Model to {export_path} >>> ")
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(
                sess=sess, 
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={'predict': prediction_signature}
            )
            builder.save()

        print(" <<< Export History >>>")
        with open(os.path.join(output_directory, "history.pkl"), "wb") as f:
            pickle.dump({
                "history": history.history,
                "auroc": auroc.aurocs,
            }, f)
        print(" <<< Export Complete! >>> ")

    finally:
        utility.delete_training_lock(output_directory)

if __name__ == "__main__":
    main()