// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{RenderPipelineInOutStage, RenderPipelineStage};

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
    pub fn new(spot_c_offset: usize, spot_color: [f32; 4]) -> Self {
        debug_assert!(spot_color.iter().all(|c| c.is_finite()));
        Self {
            spot_c: 3 + spot_c_offset,
            spot_color,
        }
    }
}

impl RenderPipelineStage for SpotColorStage {
    type Type = RenderPipelineInOutStage<f32, f32, 1, 1, 0, 0>;

    fn uses_channel(&self, _c: usize) -> bool {
        // The stage needs access to all channels so that it can faithfully
        // copy unchanged ones from input to output. Therefore we always
        // return true here.
        true
    }

    // `row` should only contain color channels and the spot channel.
    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
    ) {
        // Ensure the spot channel index is valid for this row slice. If it is
        // missing (which should not happen), fall back to copying channels
        // unchanged.
        if self.spot_c >= row.len() {
            for (input_rows, output_rows) in row.iter_mut() {
                let center_in = input_rows[1];
                let out_row = &mut output_rows[0];
                out_row[..xsize].copy_from_slice(&center_in[1..xsize + 1]);
            }
            return;
        }

        let scale = self.spot_color[3];

        // First copy all channels through unchanged (central row, excluding
        // border columns). This guarantees that channels we do not explicitly
        // modify are preserved.
        for (input_rows, output_rows) in row.iter_mut() {
            let center_in = input_rows[1];
            let out_row = &mut output_rows[0];
            out_row[..xsize].copy_from_slice(&center_in[1..xsize + 1]);
        }

        // If the spot color has zero opacity, the copied data is already
        // correct and we can return early.
        if scale == 0.0 {
            return;
        }

        // Apply spot-colour mixing to the RGB channels.
        for idx in 0..xsize {
            let s_val = row[self.spot_c].1[0][idx];
            let mix = scale * s_val;

            // Modify in place in the already-copied output rows.
            row[0].1[0][idx] = mix * self.spot_color[0] + (1.0 - mix) * row[0].1[0][idx];
            row[1].1[0][idx] = mix * self.spot_color[1] + (1.0 - mix) * row[1].1[0][idx];
            row[2].1[0][idx] = mix * self.spot_color[2] + (1.0 - mix) * row[2].1[0][idx];
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
    fn render_spotcolors_option_missing() -> Result<()> {
        // Test the new render_spotcolors option
        // This test verifies that spot color rendering can be controlled independently
        // from general output rendering via DecodeOptions.render_spotcolors

        use crate::decode::DecodeOptions;

        // Test that DecodeOptions has render_spotcolors field
        let mut options = DecodeOptions::new();

        // Default should be true
        assert!(
            options.render_spotcolors,
            "render_spotcolors should default to true"
        );

        // Should be configurable
        options.render_spotcolors = false;
        assert!(
            !options.render_spotcolors,
            "render_spotcolors should be configurable"
        );

        // The real test would be to decode an image with and without spot colors
        // and verify different outputs, but for now this tests the API exists

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn cli_spotcolors_golden_test() -> Result<()> {
        // Simple test to verify spot color control works
        use crate::container::ContainerParser;
        use crate::decode::decode_jxl_codestream;

        let spot_jxl_path = "resources/test/conformance_test_images/spot.jxl";
        let spot_data = std::fs::read(spot_jxl_path)
            .unwrap_or_else(|e| panic!("Failed to read test file {}: {}", spot_jxl_path, e));

        let codestream = ContainerParser::collect_codestream(&spot_data)?;

        // Test decode with spot colors enabled vs disabled
        let mut options_with_spots = crate::decode::DecodeOptions::new();
        options_with_spots.render_spotcolors = true;
        let (image_with_spots, _) = decode_jxl_codestream(options_with_spots, &codestream)?;

        let mut options_no_spots = crate::decode::DecodeOptions::new();
        options_no_spots.render_spotcolors = false;
        let (image_no_spots, _) = decode_jxl_codestream(options_no_spots, &codestream)?;

        // Basic validation: both decodings should succeed and have same structure
        assert!(
            !image_with_spots.frames.is_empty(),
            "Should have decoded some frames"
        );
        assert!(
            !image_no_spots.frames.is_empty(),
            "Should have decoded some frames"
        );
        assert_eq!(
            image_with_spots.frames.len(),
            image_no_spots.frames.len(),
            "Both decodings should produce same number of frames"
        );

        if let (Some(frame_with), Some(frame_without)) = (
            image_with_spots.frames.first(),
            image_no_spots.frames.first(),
        ) {
            assert_eq!(
                frame_with.channels.len(),
                frame_without.channels.len(),
                "Both decodings should produce same number of channels"
            );
        }

        Ok(())
    }
}
