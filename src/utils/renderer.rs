use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use derive_new::new;

/// A Simple renderer for TUI-disabled modes
#[derive(new)]
pub struct Simple {}

#[allow(clippy::dbg_macro)]
impl MetricsRenderer for Simple {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        dbg!(item);
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        dbg!(item);
    }
}
