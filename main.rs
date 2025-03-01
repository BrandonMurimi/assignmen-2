
mod interface;
mod model ;
mod training;
mod data ;
use crate ::model:: LinearRegressionModelConfig;
use crate::training:: *;
use {
{Autodiff, Wgpu},
Dataset,
AdamConfig,
};
fn main() {
type MyBackend = Autodiff<Wgpu<f32>>;
let device = Default::default();
let model_config = LinearRegressionModelConfig {
input_size: 1,
output_size: 1,
};
let linear_model = model_config.init::<MyBackend>(&device);
println!("{:?}", linear_model);
let training_config = LinearRegressionTrainingConfig {
model: model_config,
optimizer: AdamConfig::new(),
num_epochs: 10,
batch_size: 64,
num_workers: 4,
seed: 42,
learning_rate: 1.0e-4,
};
let artifact_dir = "artifacts";
create_artifact_dir_for_linear(artifact_dir);
// TRAINING AND INFERENCE LEFT FOR CURIOUS STUDENTS AS AN EXRCISE
train_linear_regression::<MyBackend>(
artifact_dir, training_config, device
);
// Generate synthetic data for prediction
let (_, valid_data) = generate_linear_data();
// Make predictions
inference
model
training
data
::model::
::training::
burn::
backend::
data::dataset::
optim::
data::
predict(&linear_model, &valid_data, device);
}
data.rs
use Dataset;
use {
Batcher,
*,
};
use Rng;
#[derive(Clone, Debug)]
pub struct LinearRegressionBatch<B: Backend> {
pub features: Tensor<B, 2>,
pub targets: Tensor<B, 2>,
}
#[derive(Clone)]
pub struct LinearRegressionBatcher<B: Backend> {
device: B::Device,
}
impl<B: Backend> LinearRegressionBatcher<B> {
pub fn new(device: B::Device) -> Self {
Self { device }
}
}
impl<B: Backend> Batcher<(f32, f32), LinearRegressionBatch<B>>
for LinearRegressionBatcher<B> {
fn batch(&self, items: Vec<(f32, f32)>) -> LinearRegressionBatch<B> {
let features: Vec<f32> = items.iter().map(|(x, _)| *x).collect();
let targets: Vec<f32> = items.iter().map(|(_, y)| *y).collect();
let features = Tensor::<B, 2>::from_data(
features.as_slice(), &self.device)
.reshape([items.len(), 1]);
let targets = Tensor::<B, 2>::from_data(
targets.as_slice(), &self.device)
.reshape([items.len(), 1]);
LinearRegressionBatch { features, targets }
}
}
pub struct SyntheticDataItem {
pub features: [f32; 1],
pub target: f32,
}
pub struct SyntheticDataset {
burn::data::dataset::
burn::
data::dataloader::batcher::
prelude::
rand::
data: Vec<SyntheticDataItem>,
}
impl SyntheticDataset {
pub fn new(data: Vec<SyntheticDataItem>) -> Self {
Self { data }
}
pub fn iter(&self) -> Iter<SyntheticDataItem> {
self.data.iter()
}
}
pub fn generate_linear_data() -> (SyntheticDataset, SyntheticDataset) {
let mut rng = thread_rng();
let train_data: Vec<SyntheticDataItem> = (0..1000)
.map(|_| {
let x: f32 = rng.gen();
let y: f32 = 2.0 * x + 1.0;
SyntheticDataItem {
features: [x],
target: y,
}
})
.collect();
let valid_data: Vec<SyntheticDataItem> = (0..200)
.map(|_| {
let x: f32 = rng.gen();
let y: f32 = 2.0 * x + 1.0;
SyntheticDataItem {
features: [x],
target: y,
}
})
.collect();
(
SyntheticDataset::new(train_data),
SyntheticDataset::new(valid_data),
)
}
impl Dataset<(f32, f32)> for SyntheticDataset {
fn get(&self, index: usize) -> Option<(f32, f32)> {
self.data.get(index).map(|item| (item.features[0], item.target))
}
std::slice::
rand::
fn len(&self) -> usize {
self.data.len()
}
}
model.rs
use {
{
Linear, LinearConfig,
},
*,
};
#[derive(Module, Debug)]
pub struct LinearRegressionModel<B: Backend> {
linear: Linear<B>,
}
#[derive(Config, Debug)]
pub struct LinearRegressionModelConfig {
pub input_size: usize,
pub output_size: usize,
}
impl LinearRegressionModelConfig {
/// Returns the initialized linear regression model.
pub fn init<B: Backend>(
&self, device: &B::Device) -> LinearRegressionModel<B> {
LinearRegressionModel {
linear: LinearConfig::new(
self.input_size, self.output_size
).init(device),
}
}
}
impl<B: Backend> LinearRegressionModel<B> {
/// Forward pass for the linear regression model.
/// # Arguments
/// * `input` - A tensor of shape [batch_size, input_size]
/// # Returns
/// * A tensor of shape [batch_size, output_size]
pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
self.linear.forward(input)
}
}
training.rs
burn::
nn::
prelude::
use crate generate_linear_data;
use crate::{
{LinearRegressionBatch, LinearRegressionBatcher},
{LinearRegressionModel, LinearRegressionModelConfig},
};
use {
DataLoaderBuilder,
AdamConfig,
*,
CompactRecorder,
AutodiffBackend,
*,
};
impl<B: AutodiffBackend> LinearRegressionModel<B> {
pub fn forward_regression(
&self,
features: Tensor<B, 2>,
targets: Tensor<B, 2>,
) -> RegressionOutput<B> {
let predictions = self.forward(features);
let loss = (predictions.clone() - targets.clone())
.powf(predictions.clone())
.mean();
RegressionOutput::new(loss, predictions, targets)
}
}
impl<B: AutodiffBackend>
TrainStep<LinearRegressionBatch<B>, RegressionOutput<B>>
for LinearRegressionModel<B>
{
fn step(
&self, batch: LinearRegressionBatch<B>
) -> TrainOutput<RegressionOutput<B>> {
let output =
self.forward_regression(batch.features, batch.targets);
TrainOutput::new(self, output.loss.backward(), output)
}
}
impl<B: AutodiffBackend>
ValidStep<LinearRegressionBatch<B>, RegressionOutput<B>>
for LinearRegressionModel<B>
{
fn step(
&self, batch: LinearRegressionBatch<B>
) -> RegressionOutput<B> {
::data::
data::
model::
burn::
data::dataloader::
optim::
prelude::
record::
tensor::backend::
train::metric::
self.forward_regression(batch.features, batch.targets)
}
}
#[derive(Config)]
pub struct LinearRegressionTrainingConfig {
pub model: LinearRegressionModelConfig,
pub optimizer: AdamConfig,
#[config(default = 10)]
pub num_epochs: usize,
#[config(default = 64)]
pub batch_size: usize,
#[config(default = 4)]
pub num_workers: usize,
#[config(default = 42)]
pub seed: u64,
pub learning_rate: f64,
}
remove_dir_all(artifact_dir).ok();
create_dir_all(artifact_dir).ok();
}
pub fn train_linear_regression<B: AutodiffBackend<InnerBackend = B>>(
artifact_dir: &str,
config: LinearRegressionTrainingConfig,
device: B::Device,
) where
<B as AutodiffBackend>::InnerBackend: AutodiffBackend,
{
config
.save(format!("{artifact_dir}/config.json"))
.expect("Config should be saved successfully");
let batcher_valid =
LinearRegressionBatcher::<B::InnerBackend>::new(device.clone());
let (train_data, valid_data) = generate_linear_data();
let dataloader_train = DataLoaderBuilder::new(batcher_valid.clone())
.batch_size(config.batch_size)
.shuffle(config.seed)
.num_workers(config.num_workers)
.build(train_data);
let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
.batch_size(config.batch_size)
.shuffle(config.seed)
.num_workers(config.num_workers)
std::fs::
std::fs::
.build(valid_data);
let learner = LearnerBuilder::new(artifact_dir)
.metric_train_numeric(LossMetric::new())
.metric_valid_numeric(LossMetric::new())
.with_file_checkpointer(CompactRecorder::new())
.devices(vec![device.clone()])
.num_epochs(config.num_epochs)
.summary()
.build(
config.model.init::<B>(&device),
config.optimizer.init(),
config.learning_rate,
);
let model_trained = learner.fit(dataloader_train, dataloader_valid);
model_trained
.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
.expect("Trained model should be saved successfully");
}
src/inference.rs
use crate {LinearRegressionModel, LinearRegressionModelConfig};
use crate {SyntheticDataset, generate_linear_data};
use {
*,
CompactRecorder,
AutodiffBackend,
};
pub fn load_model<B: AutodiffBackend>(
artifact_dir: &str, device: B::Device
) -> LinearRegressionModel<B> {
let model_config: LinearRegressionModelConfig =
LinearRegressionModelConfig::load(
format!("{artifact_dir}/config.json"))
.expect("Config should be loaded successfully");
let model = model_config.init::<B>(&device);
model.clone()
.load_file(
format!(
"{artifact_dir}/model"), &CompactRecorder::new(), &device
)
.expect("Model should be loaded successfully");
model
}
pub fn predict<B: AutodiffBackend>(
model: &LinearRegressionModel<B>,
dataset: &SyntheticDataset,
device: B::Device) {
for item in dataset.iter() {
let features = Tensor::<B, 2>::from_data(
item.features.as_slice(), &device
).reshape([1, 1]);
let prediction = model.forward(features);
println!(
"Features: {:?}, Prediction: {:?}",
item.features, prediction);
}
}
::model::
::data::
burn::
prelude::
record::
tensor::backend::
