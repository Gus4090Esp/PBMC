from load_state import *
from load_plotf import *
from load_data import *

## variables defined in BenchMark are needed
## for a few functions for that reason we
## create a python script that will load
## the necessary libraries/data
## also we create a script that will hlp us
## plotting our data

def warn(*args, **kwargs):
    pass

warnings.warn = warn;

## To begin we will implement a VAE
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs;
        batch = tf.shape(z_mean)[0];
        dim = tf.shape(z_mean)[1];
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)); return z_mean + tf.exp(0.5 * z_log_var) * epsilon;

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs);
        self.encoder = encoder;
        self.decoder = decoder;
        self.total_loss_tracker = keras.metrics.Mean(name = "total_loss");
        self.reconstruction_loss_tracker = keras.metrics.Mean(name = "reconstruction_loss");
        self.kl_loss_tracker = keras.metrics.Mean(name = "kl_loss");
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ];
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data);
            reconstruction = self.decoder(z);
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction)
                )
            );
            kl_loss = -.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var));
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis = 1));
            total_loss = reconstruction + kl_loss;
        grads = tape.gradient(total_loss, self.trainable_weights);
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights));
        self.total_loss_tracker.update_state(total_loss);
        self.reconstruction_loss_tracker.update_state(reconstruction_loss);
        self.kl_loss_tracker.update_state(kl_loss);
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        };

latent_dim = 2;
input_shape = len(X[0]);

## defining our architecture
encoder_input = keras.Input(shape = (input_shape,));
layer_1 = layers.Dense(64, activation = 'relu')(encoder_input);
layer_2 = layers.Dense(32, activation = 'relu')(layer_1);
layer_3 = layers.Dense(16, activation = 'relu')(layer_2);
z_mean = layers.Dense(latent_dim, name = 'z_mean')(layer_3);
z_log_var = layers.Dense(latent_dim, name = 'z_log_var')(layer_3);
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name = 'encoder');

## now lets build up the decoder portion
latent_input = keras.Input(shape=(latent_dim,));
layer_4 = layers.Dense(16, activation="relu")(latent_input);
layer_5 = layers.Dense(32, activation="relu")(layer_4);
layer_6 = layers.Dense(64, activation="relu")(layer_5);
decoder_output = layers.Dense(input_shape, activation = "sigmoid")(layer_6);
decoder = keras.Model(latent_input, decoder_output, name = "decoder");

## now we can finally apply VAE class onto our data
vae = VAE(encoder, decoder);
vae.compile(optimizer = "adam");
bb_size = int(len(X_train)/12)
N_iter = 12;
vae.fit(X_train,epochs = N_iter, batch_size = bb_size);
vae_loss = vae.history.history["loss"];
z_mean, _, _ = vae.encoder.predict(X);
## lets just make sure that nothing we haven't defined
## is in the background to make sure its all smooth
## when we implement different NN
## right now we are not going to worry about saving
## any of the weights or neural net architecture
## as we are not going to need it in the succeeding
## sessions

## We will save the figures so we can plot pretty
## things and show how to present the data
CCVAE_title = "Categorizing Cell Types Using A Variational Autoencoder";
CCVAE_path = fig_file_path + "Categorizing_Cell_Types_VAE"
xlab = "Dimension 1";
ylab = "Dimension 2";
vae_z_mean = z_mean;
keras.backend.clear_session();
## Something to note is that this variational autoencoder
## quickly overfits our data, its very efficient but the
## batch number should be small, along with the number
## epochs. In order to perform better we should use
## stochastic gradient descent and more data poitns
## I do not have more data poitns so lmao rip



## lets implement another autoencoder but this time a simple one
encoder_input = keras.Input(shape = (input_shape,), name = "img");
layer_1 = layers.Dense(64, activation = "relu")(encoder_input);
layer_2 = layers.Dense(32, activation = "relu")(layer_1);
layer_3 = layers.Dense(16, activation = "relu")(layer_2);
layer_4 = layers.Dense(4, activation = "relu")(layer_3);
bottleneck = layers.Dense(2, activation = "sigmoid", name = "bottleneck")(layer_4);
encoder = keras.Model(encoder_input, bottleneck, name = "encoder");
layer_5 = layers.Dense(4, activation = "relu")(bottleneck);
layer_6 = layers.Dense(16, activation = "relu")(layer_5);
layer_7 = layers.Dense(32, activation = "relu")(layer_6);
layer_8 = layers.Dense(64, activation = "relu")(layer_7);
decoder_output = layers.Dense(input_shape, activation = "sigmoid")(layer_8);
decoder = keras.Model(layer_5, decoder_output, name = "decoder");
autoencoder_1 = keras.Model(encoder_input, decoder_output);
autoencoder_1.compile(loss = 'MSE', optimizer = 'adam');
autoencoder_1.fit(X_train, X_train, epochs = N_iter, batch_size = bb_size);
CCA1_title = "Categorizing Cell Types Using AutoEncoder";
CCA1_path = fig_file_path + "Categorizing_Cell_Types_AE";
zz = encoder.predict(X);
ae_zz = zz;
A1_loss = autoencoder_1.history.history["loss"];
keras.backend.clear_session();

ae_zz_name = "ae_zz"
A1_loss_name = "A1_loss";
vae_z_mean_name  = "vae_z_mean";
vae_loss_name = "vae_loss";


my_names = [ae_zz_name, A1_loss_name, vae_z_mean_name, vae_loss_name];
my_data = [ae_zz, A1_loss, vae_z_mean, vae_loss];
data_name = "Data"
data_path = data_name + "/";
if not os.path.exists(data_path):
    os.system("mkdir " + data_name);

nn = len(my_names);
for i in range(nn):
    dpath = data_path + my_names[i];
    if os.path.exists(dpath):
        os.system("rm " + dpath);
    np.save(dpath + ".npy", my_data[i]);
