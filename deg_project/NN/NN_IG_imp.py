import numpy as np 
import tensorflow as tf

def get_gradients(model, sample_inputs, target_range=None, jacobian=False):
    """Computes the gradients of outputs w.r.t input.

    Args:
        sample_inputs (ndarray):: model sample inputs 
        target_rtarget_range (slice)ange: Range of target 

    Returns:
        Gradients of the predictions w.r.t input
    """
    if isinstance(sample_inputs, list):
        for i in range(len(sample_inputs)):
            sample_inputs [i] = tf.cast(sample_inputs[i], tf.float32)
    else:
        sample_inputs = tf.cast(sample_inputs, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(sample_inputs)
        preds = model(sample_inputs)
        if (target_range is None):
            target_preds = preds[:, :]
        else:
            target_preds = preds[:, target_range]

    if(jacobian):
        grads = tape.jacobian(target_preds, sample_inputs)
    else:
        grads = tape.gradient(target_preds, sample_inputs)
    return grads

def linearly_interpolate(model, sample_input, baseline=None, num_steps=50, multiple_samples=False):
    # If baseline is not provided, start with a zero baseline
    # having same size as the sample input.
    if baseline is None:
        baseline = np.zeros(sample_input.shape).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    #Do interpolation.
    sample_input = sample_input.astype(np.float32)
    interpolated_sample_input = [
        baseline + (step / num_steps) * (sample_input - baseline)
        for step in range(num_steps + 1)
    ]

    interpolated_sample_input = np.array(interpolated_sample_input).astype(np.float32)
    if(multiple_samples):
        old_shape = interpolated_sample_input.shape
        new_transpose_form = (1,0)+tuple(range(interpolated_sample_input.ndim)[2:]) #switch the two first axises
        new_shape_form = (old_shape[0]*old_shape[1],) + old_shape[2:]
        interpolated_sample_input = interpolated_sample_input.transpose(new_transpose_form).reshape(new_shape_form)

    return interpolated_sample_input, sample_input, baseline 

    
def get_integrated_gradients(model, sample_inputs, target_range=None, baselines=None, num_steps=50, multiple_samples=False):
    """Computes Integrated Gradients for range of labels.

    Args:
        model (tensorflow model): Model
        sample_inputs (ndarray): Original sample input to the model 
        target_range (slice): Target range - grdient of Target range  with respect to the input
        baseline (ndarray): The baseline to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        integrated_grads_list : Integrated gradients w.r.t input
        hypothetical_importance_list : hypothetical importance w.r.t input
    """
    #tf.compat.v1.enable_v2_behavior()
    if isinstance(sample_inputs, list):
        num_of_inputs_types = len(sample_inputs)
    else:
        #insert the inputs to a list to fit generalized code
        num_of_inputs_types = 1
        sample_inputs = [sample_inputs]
        if (baselines is not None):
            baselines = [baselines]
    
    # 1. Do interpolation.
    output = []
    if(baselines is None):
        for sample_input in sample_inputs:
            output.append(linearly_interpolate(model, sample_input, baselines, num_steps=num_steps, multiple_samples=multiple_samples))
    else:
        for sample_input, baseline in zip(sample_inputs, baselines):
            output.append(linearly_interpolate(model, sample_input, baseline, num_steps=num_steps, multiple_samples=multiple_samples))

    interpolated_samples_inputs = [x[0] for x in output]
    sample_inputs = [x[1] for x in output]
    baselines = [x[2] for x in output]

    # 2. Get the gradients
    if (num_of_inputs_types == 1):
        grads_values = get_gradients(model, interpolated_samples_inputs[0], target_range=target_range)
        grads_list = [tf.convert_to_tensor(grads_values, dtype=tf.float32)]
    else:
        grads_list = get_gradients(model, interpolated_samples_inputs, target_range=target_range)
        grads_list = [tf.convert_to_tensor(grads, dtype=tf.float32) for grads in grads_list]

    if(multiple_samples):
        num_of_samples = sample_inputs[0].shape[0]
        grads_list = [tf.reshape(grads, [num_of_samples, num_steps+1] + grads.shape.as_list()[1:]) for grads in grads_list]

    # 3. Approximate the integral using the trapezoidal rule
    if(multiple_samples):
        grads_list = [(grads[:, :-1] + grads[:,1:]) / 2.0 for grads in grads_list]
        avg_grads_list = [tf.reduce_mean(grads, axis=1) for grads in grads_list]
    else: 
        grads_list = [(grads[:-1] + grads[1:]) / 2.0 for grads in grads_list]
        avg_grads_list = [tf.reduce_mean(grads, axis=0) for grads in grads_list]  

    # 4. get hypothetical importance score - it's the average gradient
    hypothetical_importance_list = [avg_grads.numpy() for avg_grads in avg_grads_list]

    # 5. Calculate integrated gradients and return
    integrated_grads_list = [(sample_inputs[i] - baselines[i]) * avg_grads_list[i] for i in range(num_of_inputs_types)]
    integrated_grads_list = [integrated_grads.numpy() for integrated_grads in integrated_grads_list]
    if (num_of_inputs_types == 1):
        return integrated_grads_list[0], hypothetical_importance_list[0]
    return integrated_grads_list, hypothetical_importance_list


