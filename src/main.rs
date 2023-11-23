use lib::{network::Network, activations::SIGMOID, images::Image1d};
use std::io::Error;
use std::fs::File;
use std::env;
use std::time::Instant;

pub mod lib;
fn main() {
    let start = Instant::now();
    let mut inputs = vec![];
    let mut outputs = vec![];
    for n in 0..4 {
        let img0 = Image1d::grayscale_from(format!("src/lib/0{}.png", n.to_string()).as_str());
        let img1 = Image1d::grayscale_from(format!("src/lib/1{}.png", n.to_string()).as_str());
        inputs.push(img0.pixels);
        outputs.push(vec![0.0]);
        inputs.push(img1.pixels);
        outputs.push(vec![1.0]);
    }
    let mut network = Network::import("test.json");
    println!("Time to create network: {:?}", start.elapsed());

    network.train(inputs,outputs,10000);
    println!("Time to create network and train: {:?}", start.elapsed());
    network.export("test.json");
    let img = Image1d::grayscale_from("3.png");
    println!("prediction: {:?}", network.feed_forward(img.pixels));
}