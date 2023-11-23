use std::vec;
use std::fs::File;
use std::io::{BufReader, Write};
use rand::{thread_rng,Rng};
use serde_json;
use serde::{Serialize,Deserialize};

use super::matrix::Matrix;
use super::activations::{Activation, SIGMOID};


pub struct Network <'a>{
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
    activationS: String
}
#[derive(Deserialize,Serialize)]
struct Import {
    layers: Vec<usize>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
    data: Vec<Vec<Vec<f64>>>,
    learning_rate: f64,
    activation: String
}

impl Network<'_> {
    pub fn new<'a>(layers: Vec<usize>, learning_rate: f64, activation: Activation<'a>) -> Network<'a> {
        let mut weights = vec![];
        let mut biases = vec![];
            for i in 0..layers.len()-1 {
                weights.push(Matrix::random(layers[i + 1],layers[i]));
                biases.push(Matrix::random(layers[i + 1], 1));
            }
            let activationS = "sigmoid".to_string();
        Network {
            layers,weights,biases,data:vec![],learning_rate,activation,activationS
        }
    }
    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64>{
        if inputs.len() != self.layers[0] {
            panic!("Invalid number of inputs");
        }
        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() - 1 {
            current = self.weights[i].multiply(&current).add(&self.biases[i]).map(self.activation.function);
            self.data.push(current.clone());
        }
        current.data[0].to_owned()
    }
    pub fn back_propogate(&mut self, outputs: Vec<f64>, targets: Vec<f64>,learning_rate: f64) {
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets!");
        }
        let mut parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).subtract(&parsed);
        let mut gradients = parsed.map(self.activation.derivative);
        for i in (0..self.layers.len()-1).rev() {
            gradients = gradients.dot_product(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);
            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }
    pub fn stoc(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, learning_rate: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = thread_rng();
        let range_end = inputs.len() as f64*0.75; //encourage larger batches by not letting the start be further than 3/4 of the way into the data
        let range_min = inputs.len() as f64*0.05; //range must be at least 5% of dataset
        let start = rng.gen_range(0..range_end.floor() as usize);
        let end = rng.gen_range(start + range_min.ceil() as usize..inputs.len());
        let o = inputs.clone()[start..end].to_vec(); 
        let t = targets.clone()[start..end].to_vec();
        (o,t)
    }
    pub fn shuffle(mut inputs: Vec<Vec<f64>>,mut targets: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut outputs = vec![];
        let mut newTargets = vec![];
        let mut rng = thread_rng();
        let ilen = inputs.len();
        while outputs.len() < ilen {
            let index = rng.gen_range(0..inputs.len()); //inputs.len gets shorter every time, as long as we call this every time it'll stay within the range
            outputs.push(inputs.swap_remove(index)); //we don't care about ordering because we're shuffling anyway, so we use swap_remove bc it has low time complexity
            newTargets.push(targets.swap_remove(index));
        }
        (outputs,newTargets)
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
        let mut learning_rate = self.learning_rate;
        for i in 1..=epochs {
            if epochs < 1000 || i % (epochs/10) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }
            let shuffled = Network::shuffle(inputs.clone(), targets.clone());
            let (mut batch, mut batch_targets) = self.stoc(shuffled.0,shuffled.1,learning_rate);
            for j in 0..batch.len() {
                let outputs = self.feed_forward(batch[j].clone());
                self.back_propogate(outputs, batch_targets[j].clone(), learning_rate);
            }
            if learning_rate > 0.0001 && learning_rate > 0.01* self.learning_rate {
                learning_rate = learning_rate * 0.90;
            }
        }
    }


    //save and load model
    pub fn export(&mut self, filepath: &str) {
        //lol the import isn't very good if i can't figure out an export huh
        let mut file = File::create(filepath).unwrap();
        let mut weights = vec![];
        let mut biases = vec![];
        for i in 0..self.layers.len()-1 {
            weights.push(self.weights[i].data.clone());
            biases.push(self.biases[i].data.clone());
            //see if it works without data?
        }
        let export = Import {
            weights, biases, layers: self.layers.clone(), data: vec![], learning_rate: self.learning_rate, activation: self.activationS.clone()
        };
        let string = serde_json::to_string(&export).unwrap();
        let _ = file.write_all(string.as_bytes());
    }
    pub fn import<'a>(filepath: &str) -> Network<'a> {
        let file = File::open(filepath).unwrap();
        let reader = BufReader::new(file);
        let import: Import = serde_json::from_reader(reader).unwrap();
        let mut activation;
        match import.activation.as_str() {
            "sigmoid" => activation = SIGMOID,
            _=> panic!("No recognized activation function!")
        }
        let mut weights = vec![];
        let mut biases = vec![];
        for i in 0..import.layers.len()-1 {
            weights.push(Matrix::from(import.weights[i].clone()));
            biases.push(Matrix::from(import.biases[i].clone()));
            //see if it works without data?
        }
        Network { layers: import.layers, weights, biases, data: vec![], learning_rate: import.learning_rate, activation, activationS: import.activation }
    }
}