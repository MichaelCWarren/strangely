use std::io::Read;
use std::path::Path;
use std::{ffi::OsStr, io::Cursor};

use axum::{extract::Query, http::header, response::IntoResponse, routing::get, Router};
use clap::{ArgAction, Parser};
use image::{
    imageops::{overlay, FilterType},
    DynamicImage,
};

use rand::random;
use rustface::{read_model, ImageData};
use serde_derive::Deserialize;
use uuid::Uuid;

const DEFAULT_SCALE: f32 = 0.55_f32;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, required_unless_present_any=["url", "web_server"])]
    path: Option<String>,

    #[arg(short, long, required_unless_present_any=["path", "web_server"])]
    url: Option<String>,

    #[arg(short, long, default_value_t = DEFAULT_SCALE)]
    scale: f32,

    #[arg(short, long, action = ArgAction::SetTrue, required_unless_present_any=["url", "path"])]
    web_server: bool,

    #[arg(long, default_value_t = 8080)]
    port: usize,

    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    _args: Vec<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if args.web_server {
        web(args).await;
    } else {
        local(args);
    }
}

async fn web(args: Args) {
    let app = Router::new().route("/", get(root));
    let server_listening_on = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&server_listening_on)
        .await
        .unwrap();
    println!("Listening on: {server_listening_on}");
    axum::serve(listener, app).await.unwrap();
}

#[derive(Debug, Deserialize)]
struct RootParams {
    url: String,
    scale: Option<f32>,
}

async fn root(query: Option<Query<RootParams>>) -> impl IntoResponse {
    if let Some(query) = query {
        let Query(params) = query;
        let (mut img, stem, _) = get_url_image(params.url);

        strangify(&mut img, params.scale.unwrap_or(DEFAULT_SCALE));

        let bytes = vec![];
        let mut cursor = Cursor::new(bytes);
        match img.write_to(&mut cursor, image::ImageFormat::Jpeg) {
            Ok(_) => {
                let headers = [
                    (header::CONTENT_TYPE, format!("image/jpeg")),
                    (
                        header::CONTENT_DISPOSITION,
                        format!("attachment; filename=\"{}.jpg\"", stem),
                    ),
                ];

                let mut out = Vec::new();
                cursor.set_position(0);
                cursor.read_to_end(&mut out).unwrap();

                return Ok((headers, out));
            }
            Err(_) => return Err("Failed to encode image"),
        };
    } else {
        Err("No query params")
    }
}

fn local(args: Args) {
    let (mut img, filename, extension) = if let Some(path) = args.path {
        let (stem, extension) = get_filename_and_extension(&path);
        let img = image::open(path).expect("No such file or directory");
        (img, stem, extension)
    } else if let Some(url) = args.url {
        get_url_image(url)
    } else {
        panic!("no image available");
    };

    strangify(&mut img, args.scale);

    let id = Uuid::new_v4();
    let output_filename = format!("./{}_{}.{}", filename, id.to_string(), extension);
    println!("{output_filename}");

    img.save(output_filename).unwrap();
}

fn get_url_image(url: String) -> (DynamicImage, String, String) {
    let mut bytes_reader = ureq::get(&url).call().unwrap().into_reader();
    let mut buffer = vec![];
    bytes_reader.read_to_end(&mut buffer).unwrap();

    let img = image::io::Reader::new(Cursor::new(buffer))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();

    let (stem, extension) = get_filename_and_extension(&url);
    (img, stem, extension)
}

fn strangify(img: &mut DynamicImage, scale: f32) {
    let model_bytes = include_bytes!("../model/seeta_fd_frontal_v1.0.bin");
    let model = read_model(model_bytes.as_slice()).unwrap();
    let mut detector = rustface::create_detector_with_model(model);

    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

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
        let box_w_upscale = (box_width as f32 * scale) as u32;
        let box_height = bbox.height();
        let box_h_upscale = (box_height as f32 * scale) as u32;
        let x_offset = box_w_upscale / 2;
        let y_offset = box_h_upscale / 2;

        let scaled_face = faces[face_id].resize(
            box_width + box_w_upscale,
            box_height + box_h_upscale,
            FilterType::CatmullRom,
        );

        overlay(
            img,
            &scaled_face,
            face.bbox().x() as i64 - x_offset as i64,
            face.bbox().y() as i64 - y_offset as i64,
        );
    }
}

fn get_filename_and_extension(path: &str) -> (String, String) {
    let file_path = Path::new(&path);
    let stem = file_path
        .file_stem()
        .expect("file does not have a stem")
        .to_os_string()
        .into_string()
        .unwrap();
    let extension = file_path
        .extension()
        .unwrap_or(OsStr::new("img"))
        .to_os_string()
        .into_string()
        .unwrap();

    (stem, extension)
}
