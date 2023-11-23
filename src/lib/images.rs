use image::{self, GenericImageView, Rgba};

use super::matrix::Matrix;

#[derive(Debug)]
pub struct Image2d {
    pub pixels: Vec<Vec<Rgba<u8>>> //to matrix
}
pub struct Image1d {
    pub pixels: Vec<f64>,
    pub pixel_ct: usize
}
impl Image2d {
    pub fn from(filepath: &str) -> Image2d {
        let img = image::open(filepath).expect("File not found!");
        //println!("{:?}", img.pixels());
        let mut pixels = vec![vec![]];
        for pixel in img.pixels() {
            let x = pixel.0;
            let y = usize::try_from(pixel.1).unwrap();
            let rgb = pixel.2;
            if x == 0 {
                pixels.push(vec![rgb])
            }
            else {
                pixels[y].push(pixel.2);
            }
            
            //break;
        }
        Image2d {
            pixels
        }
    }
}
impl Image1d {
    pub fn from(filepath: &str) -> Image1d {
        let img = image::open(filepath).expect("File not found!");
        let dim = img.dimensions();
        let pixel_ct: usize = dim.0 as usize*dim.1 as usize;
        let mut pixels: Vec<f64> = vec![];
        for pixel in img.pixels() {
            let rgba = pixel.2.0;
            for val in rgba {
                pixels.push(val as f64/255.0);
            }
        }
        Image1d {pixels, pixel_ct}
    }
    pub fn grayscale_from(filepath: &str) -> Image1d {
        let img = image::open(filepath).expect("File not found!");
        let dim = img.dimensions();
        let pixel_ct: usize = dim.0 as usize*dim.1 as usize;
        let mut pixels: Vec<f64> = vec![];
        for pixel in img.pixels() {
            let rgba = pixel.2.0;
            let v = (rgba[0] as f64 * 0.3 + rgba[1] as f64 * 0.59 + rgba[2] as f64 * 0.11) / 255.0; //luminosity method = 0.3R + 0.59G + 0.11B. /255 to normalize
            let a = rgba[3] as f64 / 255.0; //normalize
            let value = a*v + (1.0 - a); //gives white background for alpha=0 no matter what v is, but takes v into account if 1 > alpha > 0
            //v = 0.6, a = 0.3, val= 0.6*0.3 + 0.7
            pixels.push(value)
        }

        Image1d {pixels,pixel_ct}
    }
}