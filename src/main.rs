use lib::{network::Network, activations::SIGMOID, images::Image1d};
use std::io::Error;
use std::fs::File;
use std::env;
use std::time::Instant;

pub mod lib;
fn main() {
    /*let inputs = vec![
        vec![0.0,0.0],
        vec![0.0,1.0],
        vec![1.0,0.0],
        vec![1.0,1.0],
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];*/
    let start = Instant::now();
    let mut inputs = vec![];
    let mut outputs = vec![];
    for n in 0..4 {
        let img0 = Image1d::from(format!("src/lib/0{}.png", n.to_string()).as_str());
        let img1 = Image1d::from(format!("src/lib/1{}.png", n.to_string()).as_str());
        inputs.push(img0.pixels);
        outputs.push(vec![0 as f64]);
        inputs.push(img1.pixels);
        outputs.push(vec![1 as f64]);
    }
    let mut network = Network::new(vec![inputs[0].len(),3,1],0.5,SIGMOID);
    println!("Time to create network: {:?}", start.elapsed());

    network.train(inputs,outputs,10000);
    println!("Time to create network and train: {:?}", start.elapsed());
    network.export("test.json");
    let img = Image1d::grayscale_from("3.png");
    print!("prediction: {:?}", network.feed_forward(img.pixels));
    /*let mut network = Network::import("test.json");
    println!("O,1: {:?}", network.feed_forward(vec![0.0,1.0]));
    network.train(inputs,targets,10000);
    println!("O,1: {:?}", network.feed_forward(vec![0.0,1.0]));
    network.export("test.json");*/
    //let img = Image1d::from("test.png");
    //println!("{:?}", img.pixels[0]);
    // let args: Vec<String> = env::args().collect();
    // dbg!(&args);
    // match args[1] {
        
    //     _=> ()
    // }
}
// fn loadNetwork(filepath: &str) -> Result<Network,Error>{
//     let file = File::open(filepath)?;
//     Ok(Network::import(filepath))
// }