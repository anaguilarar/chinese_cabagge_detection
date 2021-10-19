from object_detection.utils import config_util
from object_detection.builders import model_builder

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt

import random
from utils import image_processing as img_pr

import numpy as np

# %%

def save_checkpoint(chk_path, model):
    checkpoint = tf.compat.v2.train.Checkpoint(model=model)

    manager = tf.train.CheckpointManager(checkpoint, chk_path, max_to_keep=3)
    import os
    # checkpoint.step.assign_add(1)
    save_path = manager.save()


def restore_check_point(chkp_path, detection_model, first=True):
    if first:
        tmp_box_predictor_checkpoint = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )

        tmp_model_checkpoint = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=tmp_box_predictor_checkpoint)

        checkpoint = tf.compat.v2.train.Checkpoint(model=tmp_model_checkpoint)
        checkpoint.restore(chkp_path).expect_partial()
    else:
        checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
        checkpoint.restore(chkp_path)
    # Restore the checkpoint to the checkpoint path

    # use the detection model's `preprocess()` method and pass a dummy image
    tmp_image, tmp_shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    tmp_detections = detection_model.postprocess(tmp_prediction_dict, tmp_shapes)
    print('Weights restored!')


## models function
def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None,
                    min_score=0.6):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
          and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
          this function assumes that the boxes to be plotted are groundtruth
          boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
          category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """

    image_np_with_annotations = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=min_score)

    if image_name:
        plt.imsave(image_name, image_np_with_annotations)

    else:
        
        plt.imshow(image_np_with_annotations)


# decorate with @tf.function for faster training (remember, graph mode!)

def model_initialization(pipeline_path, classes):
    tf.keras.backend.clear_session()
    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(pipeline_path)
    # Read in the object stored at the key 'model' of the configs dictionary
    model_config = configs['model']
    # Modify the number of classes from its default of 90
    model_config.ssd.num_classes = classes
    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = True

    model = model_builder.build(
        model_config=model_config, is_training=True)

    return model


# Again, uncomment this decorator if you want to run inference eagerly

def detect(input_tensor, model):
    """Run detection on an input image.

    Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

    Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
    """
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)

    ### START CODE HERE (Replace instances of `None` with your code) ###
    # use the detection model's postprocess() method to get the the final detections
    detections = model.postprocess(prediction_dict, shapes)
    ### END CODE HERE ###

    return detections


@tf.function
def train_step_fn(image_list,
                  groundtruth_boxes_list,
                  groundtruth_classes_list,
                  model,
                  optimizer,
                  vars_to_fine_tune,
                  batch_size):
    """A single training iteration.

    Args:
      image_list: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)

    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)

    with tf.GradientTape() as tape:
        ### START CODE HERE (Replace instances of `None` with your code) ###

        # Preprocess the images

        preprocessed_image_tensor = tf.concat(
            [model.preprocess(image_tensor)[0]
             for image_tensor in image_list], axis=0)

        true_shape_tensor = preprocessed_image_tensor.shape

        # Make a prediction
        prediction_dict = model.predict(preprocessed_image_tensor, shapes)

        # Calculate the total loss (sum of both losses)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        # Calculate the gradients
        gradients = tape.gradient(total_loss, vars_to_fine_tune)

        # Optimize the model's selected variables
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))

        ### END CODE HERE ###

    return losses_dict


def train_save(images_list,
               bb_list,
               classes_dict,
               model,
               opt,
               to_fine_tune,
               checkfolder="ckpts", n_batches=2000, batch_size=32, history=True):
    """

    :param images_list:
    :param bb_list:
    :param classes_dict:
    :param model:
    :param opt:
    :param to_fine_tune:
    :param checkfolder:
    :param n_batches:
    :param batch_size:
    :param history:
    :return:
    """
    print('Start fine-tuning!', flush=True)
    total_losslist = []
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               optimizer=opt,
                               model=model)

    manager = tf.train.CheckpointManager(ckpt,
                                         checkfolder,
                                         max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    all_keys = list(range(len(images_list)))

    for idx in range(n_batches):
        ckpt.step.assign_add(1)

        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]
        print(example_keys)
        train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors = img_pr.convert_to_tf_tensors(
            images_list[example_keys], bb_list[example_keys], classes_dict)

        # Get the ground truth
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        # get the images
        image_tensors = [train_image_tensors[key] for key in example_keys]

        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors,
                                   gt_boxes_list,
                                   gt_classes_list,
                                   model,
                                   opt,
                                   to_fine_tune,
                                   batch_size
                                   )
        toloss = total_loss['Loss/localization_loss'] + total_loss['Loss/classification_loss']
        total_losslist.append(total_loss)

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(n_batches)
                  + ', loc_loss=' + str(total_loss['Loss/localization_loss'].numpy()) +
                  ', class_loss' + str(total_loss['Loss/classification_loss'].numpy()) +
                  ', total_loss' + str(toloss.numpy()),
                  flush=True)
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        if history:
            return total_losslist


# %%
def single_image_detection(image, model,
                           cat_index,
                           filename=None,
                           min_score=0.55,
                           fig_size=(15, 20),
                           ):
    label_id_offset = 1
    tensor_img = tf.expand_dims(tf.convert_to_tensor(
        image, dtype=tf.float32), axis=0)

    detections = detect(tensor_img, model)

    plot_detections(
        image,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.uint32)
        + label_id_offset,
        detections['detection_scores'][0].numpy(),
        cat_index,
        figsize=fig_size,
        image_name=filename,
        min_score=min_score
    )

    return detections

