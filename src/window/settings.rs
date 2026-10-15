use rmpv::Value;

use crate::error_msg;
use crate::settings::*;

#[cfg(target_os = "macos")]
pub use crate::platform::macos::settings::*;

#[derive(Clone, SettingGroup, PartialEq)]
pub struct WindowSettings {
    pub confirm_quit: bool,
    pub cursor_hack: bool,
    pub fullscreen: bool,
    pub has_mouse_grid_detection: bool,
    pub hide_mouse_when_typing: bool,
    pub input_ime: bool,
    pub iso_layout: bool,
    pub normal_opacity: f32,
    #[alias = "transparency"]
    pub opacity: f32,
    pub padding_bottom: u32,
    pub padding_left: u32,
    pub padding_right: u32,
    pub padding_top: u32,
    pub refresh_rate: u64,
    pub refresh_rate_idle: u64,
    pub remember_window_position: bool,
    pub remember_window_size: bool,
    pub scale_factor: f32,
    pub show_border: bool,
    pub theme: ThemeSettings,
    pub touch_deadzone: f32,
    pub touch_drag_timeout: f32,
    pub window_blurred: bool,
    pub window_blur_radius: u32,
    pub message_area_drag_selection: bool,
    #[cfg(target_os = "windows")]
    pub corner_preference: CornerPreference,
    #[cfg(target_os = "macos")]
    pub input_macos_alt_is_meta: bool,
    #[cfg(target_os = "macos")]
    pub input_macos_option_key_is_meta: OptionAsMeta,
    #[cfg(target_os = "macos")]
    pub macos_simple_fullscreen: bool,
    #[cfg(target_os = "macos")]
    pub highlight_matching_pair: bool,
    #[cfg(target_os = "windows")]
    pub title_background_color: String,
    #[cfg(target_os = "windows")]
    pub title_text_color: String,

    #[option = "mousemoveevent"]
    pub mouse_move_event: bool,
    #[option = "lines"]
    pub observed_lines: Option<u64>,
    #[option = "columns"]
    pub observed_columns: Option<u64>,
}

impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            confirm_quit: true,
            cursor_hack: true,
            fullscreen: false,
            has_mouse_grid_detection: false,
            hide_mouse_when_typing: false,
            input_ime: true,
            iso_layout: false,
            normal_opacity: 1.0,
            opacity: 1.0,
            padding_bottom: 0,
            padding_left: 0,
            padding_right: 0,
            padding_top: 0,
            refresh_rate: 60,
            refresh_rate_idle: 5,
            remember_window_position: true,
            remember_window_size: true,
            scale_factor: 1.0,
            show_border: true,
            theme: ThemeSettings::Auto,
            touch_deadzone: 6.0,
            touch_drag_timeout: 0.17,
            window_blurred: false,
            window_blur_radius: 20,
            message_area_drag_selection: true,
            #[cfg(target_os = "windows")]
            corner_preference: CornerPreference::Default,
            #[cfg(target_os = "macos")]
            input_macos_alt_is_meta: false,
            #[cfg(target_os = "macos")]
            input_macos_option_key_is_meta: OptionAsMeta::None,
            #[cfg(target_os = "macos")]
            macos_simple_fullscreen: false,
            #[cfg(target_os = "macos")]
            highlight_matching_pair: false,
            #[cfg(target_os = "windows")]
            title_background_color: "".to_string(),
            #[cfg(target_os = "windows")]
            title_text_color: "".to_string(),

            // Neovim options
            mouse_move_event: false,
            observed_columns: None,
            observed_lines: None,
        }
    }
}

/// Higher radii render strangely and get slow, Ghostty uses the same limit.
pub const MAX_WINDOW_BLUR_RADIUS: u32 = 255;

impl WindowSettings {
    /// The OS only blurs behind the window when blur is on and something is actually translucent.
    pub fn blur_enabled(&self) -> bool {
        self.window_blurred && self.opacity.min(self.normal_opacity) < 1.0
    }

    /// The blur radius to apply, 0 meaning no blur.
    #[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
    pub fn effective_blur_radius(&self) -> u32 {
        if self.blur_enabled() { self.window_blur_radius.min(MAX_WINDOW_BLUR_RADIUS) } else { 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeSettings {
    Auto,
    Dark,
    Light,
    BgColor,
}

impl ParseFromValue for ThemeSettings {
    fn parse_from_value(&mut self, value: Value) {
        if value.is_str() {
            *self = match value.as_str().unwrap() {
                "auto" => ThemeSettings::Auto,
                "dark" => ThemeSettings::Dark,
                "light" => ThemeSettings::Light,
                "bg_color" => ThemeSettings::BgColor,
                value => {
                    error_msg!(
                        "Setting Theme expected one of `auto`, `dark`, `light`, `bg_color`, but received {value:?}"
                    );
                    return;
                }
            };
        } else {
            error_msg!("Setting Theme expected string, but received {value:?}");
        }
    }
}

impl From<ThemeSettings> for Value {
    fn from(value: ThemeSettings) -> Self {
        match value {
            ThemeSettings::Auto => Value::from("auto"),
            ThemeSettings::Dark => Value::from("dark"),
            ThemeSettings::Light => Value::from("light"),
            ThemeSettings::BgColor => Value::from("bg_color"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(target_os = "windows")]
pub enum CornerPreference {
    Default,
    Round,
    RoundSmall,
    DoNotRound,
}

#[cfg(target_os = "windows")]
impl ParseFromValue for CornerPreference {
    fn parse_from_value(&mut self, value: Value) {
        if value.is_str() {
            *self = match value.as_str().unwrap() {
                "default" => CornerPreference::Default,
                "round" => CornerPreference::Round,
                "round_small" => CornerPreference::RoundSmall,
                "do_not_round" => CornerPreference::DoNotRound,
                value => {
                    error_msg!(
                        "Setting CornerPreference expected one of `default`, `do_not_round`, `round`, `round_small`, but received {value:?}"
                    );
                    return;
                }
            };
        } else {
            error_msg!("Setting CornerPreference expected string, but received {value:?}");
        }
    }
}

#[cfg(target_os = "windows")]
impl From<CornerPreference> for Value {
    fn from(value: CornerPreference) -> Self {
        match value {
            CornerPreference::Default => Value::from("default"),
            CornerPreference::Round => Value::from("round"),
            CornerPreference::RoundSmall => Value::from("round_small"),
            CornerPreference::DoNotRound => Value::from("do_not_round"),
        }
    }
}

#[cfg(target_os = "windows")]
impl From<CornerPreference> for winit::platform::windows::CornerPreference {
    fn from(value: CornerPreference) -> Self {
        use winit::platform::windows::CornerPreference as WinitCornerPreference;

        match value {
            CornerPreference::Default => WinitCornerPreference::Default,
            CornerPreference::Round => WinitCornerPreference::Round,
            CornerPreference::RoundSmall => WinitCornerPreference::RoundSmall,
            CornerPreference::DoNotRound => WinitCornerPreference::DoNotRound,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blurred_settings(opacity: f32, normal_opacity: f32) -> WindowSettings {
        WindowSettings { window_blurred: true, opacity, normal_opacity, ..Default::default() }
    }

    #[test]
    fn blur_enabled_requires_translucency() {
        assert!(!WindowSettings { opacity: 0.9, ..Default::default() }.blur_enabled());
        assert!(!blurred_settings(1.0, 1.0).blur_enabled());
        assert!(blurred_settings(0.9, 1.0).blur_enabled());
        assert!(blurred_settings(1.0, 0.8).blur_enabled());
    }

    #[test]
    fn effective_blur_radius_defaults_to_20_and_clamps() {
        assert_eq!(blurred_settings(0.9, 1.0).effective_blur_radius(), 20);

        let settings = WindowSettings { window_blur_radius: 1000, ..blurred_settings(0.9, 1.0) };
        assert_eq!(settings.effective_blur_radius(), MAX_WINDOW_BLUR_RADIUS);

        assert_eq!(blurred_settings(1.0, 1.0).effective_blur_radius(), 0);
    }
}
