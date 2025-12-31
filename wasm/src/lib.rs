#![no_std]

extern crate alloc;
extern crate core;

mod utils;


use alloc::{vec, vec::Vec};
use core::iter::zip;
use core::ops::ControlFlow::Continue;

use wasm_bindgen::prelude::*;

use re::prelude::*;

use re::math::rand::{self, Distrib, VectorsOnUnitDisk};
use re::render::tex::{Atlas, Layout, SamplerClamp};
use re::render::{render, shader, Model, View};
use re::util::pnm::read_pnm;

use re_front::wasm::Window;

#[wasm_bindgen(start)]
fn main() {
    let res = (1024, 768);
    let mut win = Window::new(res).unwrap();

    let spark_tex = Buf2::new_with((64, 64), |x, y| {
        let x = x as f32 / 32.0 - 1.0;
        let y = y as f32 / 32.0 - 1.0;
        let d = (x * x + y * y) * 1.5;

        rgb(1.2 - d, 1.2 - d * 1.5, 1.2 - d * 3.0).to_color3()
    })
        .into();

    static FONT: &[u8] = include_bytes!("../../assets/font_16x24.pbm");
    let font = read_pnm(FONT).unwrap();
    let (cw, ch) = (font.width() / 16, font.height() / 16);
    let font = Atlas::new(Layout::Grid { sub_dims: (cw, ch) }, font.into());

    let text = b"Iloista uutta vuotta";

    let mut buf = Buf2::new((cw * text.len() as u32, ch));
    for (&c, x) in zip(text, 0..) {
        buf.slice_mut(((x * cw)..(x + 1) * cw, 0..ch))
            .copy_from(*font.get(c as u32).data());
    }
    let text = buf.into();

    let tris = [Tri([0, 1, 2]), Tri([0, 2, 3])];
    let verts = [
        (pt3(0.0, 0.0, 0.0), uv(0.0, 0.0)),
        (pt3(1.0, 0.0, 0.0), uv(1.0, 0.0)),
        (pt3(1.0, 1.0, 0.0), uv(1.0, 1.0)),
        (pt3(0.0, 1.0, 0.0), uv(0.0, 1.0)),
    ]
        .map(|(pos, uv): (Point3<Model>, _)| vertex(pos - vec3(0.5, 0.5, 0.0), uv));

    let proj =
        perspective(1.0, win.dims.0 as f32 / win.dims.1 as f32, 0.1..1000.0);
    let to_screen = viewport(pt2(0, res.1)..pt2(res.0, 0));

    let mut rng = rand::DefaultRng::from_time();

    const N: usize = 1000;

    let spline = <BezierSpline<Point2>>::new([
        // 2
        pt2(-2.0, 1.8),
        //
        pt2(-2.0, 2.1),
        pt2(-1.1, 2.1),
        //
        pt2(-1.1, 1.5),
        //
        pt2(-1.1, 1.0),
        pt2(-1.1, 1.0),
        //
        pt2(-2.0, 0.0),
        //
        pt2(-1.5, 0.0),
        pt2(-1.5, 0.0),
        //
        pt2(-1.1, 0.0),
        //
        // 0
        pt2(-1.0, 0.0),
        pt2(-1.0, 0.0),
        //
        pt2(-0.5, 0.0),
        //
        pt2(-0.8, 0.0),
        pt2(-1.0, 0.2),
        //
        pt2(-1.0, 1.0),
        //
        pt2(-1.0, 1.8),
        pt2(-0.8, 2.0),
        //
        pt2(-0.5, 2.0),
        //
        pt2(-0.2, 2.0),
        pt2(0.0, 1.8),
        //
        pt2(0.0, 1.0),
        //
        pt2(0.0, 0.2),
        pt2(-0.2, 0.0),
        //
        pt2(-0.5, 0.0),
        //
        pt2(-0.2, 0.0),
        pt2(0.0, 0.2),
        // 2
        pt2(0.0, 1.5),
        //
        pt2(0.0, 2.1),
        pt2(0.9, 2.1),
        //
        pt2(0.9, 1.5),
        //
        pt2(0.9, 1.0),
        pt2(0.9, 1.0),
        //
        pt2(0.0, 0.0),
        //
        pt2(0.5, 0.0),
        pt2(0.5, 0.0),
        //
        pt2(0.7, 0.0),
        // 6
        pt2(1.0, 0.0),
        pt2(1.0, 1.0),
        //
        pt2(1.5, 1.0),
        //
        pt2(1.8, 1.0),
        pt2(2.0, 0.8),
        //
        pt2(2.0, 0.5),
        //
        pt2(2.0, 0.2),
        pt2(1.8, 0.0),
        //
        pt2(1.5, 0.0),
        //
        pt2(1.0, 0.0),
        pt2(1.0, 0.0),
        //
        pt2(1.0, 1.0),
        //
        pt2(1.0, 2.0),
        pt2(1.8, 2.0),
        //
        pt2(2.0, 2.0),
    ]);

    #[derive(Copy, Clone, Default)]
    struct Spark {
        pos: Point2,
        vel: Vec2,
        live: f32,
        size: f32,
    }

    let mut sparks = vec![Spark::default(); N];

    win.ctx.color_clear = Some(rgba(8, 8, 32, 255));
    win.ctx.face_cull = None;
    win.ctx.depth_test = None;

    let mut s_verts = Vec::new();
    let mut s_faces = Vec::new();

    win.run(move |mut frame| {
        let t = frame.t.as_secs_f32();
        let dt = frame.dt.as_secs_f32();

        // Emit
        for spark in sparks
            .iter_mut()
            .filter(|s| s.live <= 0.0)
            .take(6)
        {
            let t = t / 4.0 + (-0.02..0.02).sample(&mut rng).clamp(0.0, 1.0);
            let emit_pos = spline.eval(t % 1.0);
            let size = (0.0..1.0).sample(&mut rng).powi(5);
            spark.live = (0.5..1.5).sample(&mut rng);
            spark.size = size * 0.1+0.05;
            let d = VectorsOnUnitDisk.sample(&mut rng);
            spark.pos = emit_pos + d * 0.02;
            spark.vel = d * 0.002;
        }
        // Update

        s_verts.clear();
        s_faces.clear();
        for spark in &mut sparks {
            if spark.live <= 0.0 {
                continue;
            }
            spark.live -= dt / 3.0;
            spark.pos += spark.vel;
            spark.vel *= 0.99;
            //spark.vel += vec2(0.0, -0.0001);

            let l = s_verts.len();
            s_faces.extend(
                tris.iter()
                    .map(|Tri([i, j, k])| tri(i + l, j + l, k + l)),
            );
            s_verts.extend(verts.iter().map(|v| {
                vertex(
                    v.pos * spark.size + spark.pos.to_vec().to_vec3().to(),
                    (v.attrib, spark.live.powf(0.2)),
                )
            }));
        }
        // Render

        let spark_shader = shader::new(
            |v: Vertex<Point3<_>, (TexCoord, f32)>, tf: ProjMat3<_>| {
                vertex(tf.apply(&v.pos), v.attrib)
            },
            |f: Frag<(TexCoord, f32)>| {
                let (uv, life) = f.var;
                let c = SamplerClamp.sample(&spark_tex, uv);
                let c = c.to_color3f() * life;
                (c.r() > 0.3).then(|| c.to_color4())
            },
        );
        let mv: Mat4<Model, View> = translate(vec3(0.0, -0.5, 0.8 * t.sin()))
            .then(&rotate_y(rads(0.5 * (t * 0.59).sin())))
            .then(&rotate_z(rads(0.5 * (t * 1.13).sin())))
            .then(&translate(vec3(0.0, -1.0, 5.0)))
            .to();
        let mvp: ProjMat3<Model> = mv.then(&proj);
        render(
            &s_faces,
            &s_verts,
            &spark_shader,
            mvp,
            to_screen,
            &mut frame.buf,
            frame.ctx,
        );

        let text_shader = shader::new(
            |v: Vertex<_, _>, mvp: &ProjMat3<_>| {
                vertex(mvp.apply(&v.pos), v.attrib)
            },
            |frag: Frag<TexCoord>| {
                let c = SamplerClamp.sample(&text, frag.var).to_rgba();
                (c.r() > 0).then(|| rgba(0xFF, 0xEE, 0x99, 0xFF))
            },
        );
        let mvp: ProjMat3<Model> = scale(vec3(4.0, -0.4, 4.0))
            .then(&translate(2.5 * Vec3::Y))
            .to()
            .then(&mv)
            .then(&proj);

        render(
            tris,
            verts,
            &text_shader,
            &mvp,
            to_screen,
            &mut frame.buf,
            &frame.ctx,
        );

        Continue(())
    });
}
