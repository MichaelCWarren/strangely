use std::path::Path;

use clap::{command, Parser};
use image::imageops::{overlay, FilterType};

use rand::random;
use rustface::{read_model, ImageData};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    image: String,
}

fn main() {
    let args = Args::parse();

    let model_bytes = include_bytes!("../model/seeta_fd_frontal_v1.0.bin");
    let model = read_model(model_bytes.as_slice()).unwrap();
    let mut detector = rustface::create_detector_with_model(model);

    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    let path = args.image.clone();
    let path = Path::new(&path);
    let mut img = image::open(args.image).unwrap();

    let width = img.width();
    let height = img.height();

    let gray = img.to_luma8();
    let mut image = ImageData::new(&gray, width, height);

    let faces = [
        image::load_from_memory(include_bytes!("../strangeway/strangeway0.png").as_slice())
            .unwrap(),
        image::load_from_memory(include_bytes!("../strangeway/strangeway1.png").as_slice())
            .unwrap(),
    ];

    for face in detector.detect(&mut image).into_iter() {
        let face_id = if random() { 1 } else { 0 };
        let bbox = face.bbox();
        let box_width = bbox.width();
        let box_w_upscale = (box_width as f32 * 0.55) as u32;
        let box_height = bbox.height();
        let box_h_upscale = (box_height as f32 * 0.55) as u32;
        let x_offset = box_w_upscale / 2;
        let y_offset = box_h_upscale / 2;

        let scaled_face = faces[face_id].resize(
            box_width + box_w_upscale,
            box_height + box_h_upscale,
            FilterType::CatmullRom,
        );

        overlay(
            &mut img,
            &scaled_face,
            face.bbox().x() as i64 - x_offset as i64,
            face.bbox().y() as i64 - y_offset as i64,
        );
    }

    let id = Uuid::new_v4();
    let stem = path
        .file_stem()
        .unwrap()
        .to_os_string()
        .into_string()
        .unwrap();
    let extension = path
        .extension()
        .unwrap()
        .to_os_string()
        .into_string()
        .unwrap();

    let output_filename = format!("./{}_{}.{}", stem, id.to_string(), extension);
    println!("{output_filename}");

    img.save(output_filename).unwrap();
}
