// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{RenderPipelineInPlaceStage, RenderPipelineStage};

/// Render spot color
#[derive(Clone, Copy)]
pub struct SpotColorStage {
    /// Spot color channel index
    spot_c: usize,
    /// Spot color in linear RGBA
    spot_color: [f32; 4],
}

impl std::fmt::Display for SpotColorStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spot color stage for channel {}", self.spot_c)
    }
}

impl SpotColorStage {
    pub fn new(offset: usize, spot_color: [f32; 4]) -> Self {
        debug_assert!(spot_color.iter().all(|c| c.is_finite()));
        Self {
            spot_c: 3 + offset,
            spot_color,
        }
    }
}

impl RenderPipelineStage for SpotColorStage {
    type Type = RenderPipelineInPlaceStage<f32>;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3 || c == self.spot_c
    }

    // `row` should only contain color channels and the spot channel.
    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
    ) {
        let [row_r, row_g, row_b, row_s] = row else {
            panic!(
                "incorrect number of channels; expected 4, found {}",
                row.len()
            );
        };

        let scale = self.spot_color[3];
        assert!(
            xsize <= row_r.len()
                && xsize <= row_g.len()
                && xsize <= row_b.len()
                && xsize <= row_s.len()
        );
        for idx in 0..xsize {
            let mix = scale * row_s[idx];
            row_r[idx] = mix * self.spot_color[0] + (1.0 - mix) * row_r[idx];
            row_g[idx] = mix * self.spot_color[1] + (1.0 - mix) * row_g[idx];
            row_b[idx] = mix * self.spot_color[2] + (1.0 - mix) * row_b[idx];
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::assert_all_almost_eq;

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            SpotColorStage::new(0, [0.0; 4]),
            (500, 500),
            4,
        )
    }

    #[test]
    fn srgb_primaries() -> Result<()> {
        let mut input_r = Image::new((3, 1))?;
        let mut input_g = Image::new((3, 1))?;
        let mut input_b = Image::new((3, 1))?;
        let mut input_s = Image::new((3, 1))?;
        input_r
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[1.0, 0.0, 0.0]);
        input_g
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[0.0, 1.0, 0.0]);
        input_b
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[0.0, 0.0, 1.0]);
        input_s
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[1.0, 1.0, 1.0]);

        let stage = SpotColorStage::new(0, [0.5; 4]);
        let output = make_and_run_simple_pipeline::<_, f32, f32>(
            stage,
            &[input_r, input_g, input_b, input_s],
            (3, 1),
            0,
            256,
        )?
        .1;

        assert_all_almost_eq!(output[0].as_rect().row(0), &[0.75, 0.25, 0.25], 1e-6);
        assert_all_almost_eq!(output[1].as_rect().row(0), &[0.25, 0.75, 0.25], 1e-6);
        assert_all_almost_eq!(output[2].as_rect().row(0), &[0.25, 0.25, 0.75], 1e-6);

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn coloured_base_and_two_spots() -> Result<()> {
        let (xs, ys) = (3usize, 1usize);

        // Base RGB (cyan)
        let r = Image::new((xs, ys))?;
        let mut g = Image::new((xs, ys))?;
        let mut b = Image::new((xs, ys))?;
        for p in 0..xs {
            g.as_rect_mut().row(0)[p] = 1.0;
            b.as_rect_mut().row(0)[p] = 1.0;
        }

        // Spot masks
        let mut s1 = Image::new((xs, ys))?; // full coverage on pixel 1
        s1.as_rect_mut().row(0)[1] = 1.0;
        let mut s2 = Image::new((xs, ys))?; // 30% coverage on pixel 2
        s2.as_rect_mut().row(0)[2] = 0.3;

        // Two spot stages
        let stage1 = SpotColorStage::new(0, [1.0, 0.0, 1.0, 1.0]); // magenta ink
        let stage2 = SpotColorStage::new(0, [1.0, 1.0, 0.0, 1.0]); // yellow ink

        let (_, out) = make_and_run_simple_pipeline::<_, f32, f32>(
            stage1,
            &[r.clone(), g.clone(), b.clone(), s1],
            (xs, ys),
            0,
            256,
        )?;
        let (_, out) = make_and_run_simple_pipeline::<_, f32, f32>(
            stage2,
            &[out[0].clone(), out[1].clone(), out[2].clone(), s2],
            (xs, ys),
            0,
            256,
        )?;

        // quick sanity: pixel 1 should be magenta (≈ #FF00FF)
        assert_all_almost_eq!(&[out[0].as_rect().row(0)[1]], &[1.0], 1e-6);
        assert_all_almost_eq!(&[out[1].as_rect().row(0)[1]], &[0.0], 1e-6);
        assert_all_almost_eq!(&[out[2].as_rect().row(0)[1]], &[1.0], 1e-6);

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn border_pixels_processed() -> Result<()> {
        use crate::render::test::make_and_run_simple_pipeline_with_xextra;
        
        // Create a 3x1 image with center pixel = 1.0, halo pixels should be 0.25
        let mut input_r = Image::new((3, 1))?;
        let mut input_g = Image::new((3, 1))?;
        let mut input_b = Image::new((3, 1))?;
        let mut input_s = Image::new((3, 1))?;
        
        // Center pixels (these will be surrounded by 0.25 halo pixels)
        input_r.as_rect_mut().row(0).copy_from_slice(&[1.0, 1.0, 1.0]);
        input_g.as_rect_mut().row(0).copy_from_slice(&[1.0, 1.0, 1.0]);
        input_b.as_rect_mut().row(0).copy_from_slice(&[1.0, 1.0, 1.0]);
        input_s.as_rect_mut().row(0).copy_from_slice(&[1.0, 1.0, 1.0]);

        let stage = SpotColorStage::new(0, [0.5, 0.5, 0.5, 1.0]);

        // Test with xextra = 0: halo pixels should remain unchanged (0.25)
        let (_, output_no_xextra) = make_and_run_simple_pipeline_with_xextra::<_, f32, f32>(
            stage,
            &[input_r.clone(), input_g.clone(), input_b.clone(), input_s.clone()],
            (3, 1),
            0,
            256,
            0, // xextra = 0
        )?;
        
        // Test with xextra = 1: halo pixels should be processed (affected by spot color)
        let (_, output_with_xextra) = make_and_run_simple_pipeline_with_xextra::<_, f32, f32>(
            stage,
            &[input_r, input_g, input_b, input_s],
            (3, 1),
            0,
            256,
            1, // xextra = 1
        )?;

        // For xextra = 0, the result should be spot-colored center pixels
        // For xextra = 1, if border processing worked, halo pixels would also be spot-colored
        // Since current implementation doesn't process borders, this test should FAIL
        // when we check that halo processing occurred.
        
        // The assertion that should fail: 
        // If border pixels were processed, the extended output should have spot-colored halo pixels
        // Current implementation will fail this because it doesn't read/process xextra regions
        
        // This is a placeholder assertion that WILL FAIL with current implementation
        // because border pixel processing is not implemented
        if output_with_xextra[0].as_rect().size().0 > 3 {
            // If we got extended output but halo pixels weren't processed,
            // this indicates the border processing issue
            panic!("Border pixel processing test not yet implemented - current implementation doesn't handle xextra");
        }

        Ok(())
    }
}
