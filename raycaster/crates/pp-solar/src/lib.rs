use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};

/// Angular radius of the solar disc in degrees (~16 arcminutes = 0.2666°).
pub const SOLAR_DISC_RADIUS_DEG: f64 = 0.2666;

/// Angular radius of the lunar disc in degrees (~15.5 arcminutes = 0.258°).
pub const LUNAR_DISC_RADIUS_DEG: f64 = 0.2583;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SolarPosition {
    /// Degrees clockwise from True North [0.0, 360.0)
    pub azimuth_deg: f64,
    /// Geometric elevation above mathematical horizon in degrees [-90.0, 90.0]
    pub geometric_elevation_deg: f64,
    /// Apparent elevation including standard atmospheric refraction in degrees
    pub apparent_elevation_deg: f64,
    /// True if apparent elevation is between -1.0° and +6.0° (Golden Hour)
    pub is_golden_hour: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LunarPosition {
    /// Degrees clockwise from True North [0.0, 360.0)
    pub azimuth_deg: f64,
    /// Topocentric geometric elevation above mathematical horizon in degrees [-90.0, 90.0]
    pub geometric_elevation_deg: f64,
    /// Apparent elevation including topocentric parallax and atmospheric refraction in degrees
    pub apparent_elevation_deg: f64,
    /// Illuminated fraction of the lunar disc [0.0, 1.0]
    pub illuminated_fraction: f64,
    /// Lunar phase elongation angle [0.0, 360.0): 0° = New, 90° = First Quarter, 180° = Full, 270° = Third Quarter
    pub phase_angle_deg: f64,
    /// Human-readable lunar phase label (e.g. "Waxing Crescent", "Full Moon")
    pub phase_name: String,
    /// Earth-Moon center distance in kilometers
    pub distance_km: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GalacticCorePosition {
    /// Azimuth of Sagittarius A* (Galactic Center) in degrees clockwise from True North [0.0, 360.0)
    pub azimuth_deg: f64,
    /// Geometric elevation in degrees [-90.0, 90.0]
    pub geometric_elevation_deg: f64,
    /// Apparent elevation including atmospheric refraction in degrees
    pub apparent_elevation_deg: f64,
    /// True when the Galactic Core is above the mathematical flat horizon (apparent_elevation_deg > 0.0)
    pub is_above_horizon: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarAlignmentWindow {
    pub date: String,
    pub start_utc: DateTime<Utc>,
    pub peak_utc: DateTime<Utc>,
    pub end_utc: DateTime<Utc>,
    /// Local solar time string (approximate or timezone-shifted for display)
    pub peak_local_solar_time: String,
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    /// Clearance above the notch floor in degrees
    pub clearance_above_floor_deg: f64,
    /// Depth below the flanking notch walls in degrees (measures beam collimation/contrast)
    pub depth_below_walls_deg: f64,
    /// Event phase ("Sunrise" or "Sunset")
    pub phase: String,
}

/// Computes high-precision solar azimuth and elevation for a given WGS84 coordinate and UTC timestamp.
/// Based on the NOAA / Meeus Astronomical Algorithms equations with Bennett atmospheric refraction.
pub fn solar_position(lat_deg: f64, lon_deg: f64, dt: DateTime<Utc>) -> SolarPosition {
    let jd = julian_day(dt);
    let jc = (jd - 2451545.0) / 36525.0; // Julian century from J2000.0

    // Geometric mean longitude of the sun (deg)
    let l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)).rem_euclid(360.0);
    // Geometric mean anomaly of the sun (deg)
    let m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
    let m_rad = m.to_radians();

    // Eccentricity of Earth's orbit
    let e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);

    // Sun equation of center
    let c = m_rad.sin() * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + (2.0 * m_rad).sin() * (0.019993 - 0.000101 * jc)
        + (3.0 * m_rad).sin() * 0.000289;

    // Sun true longitude & apparent longitude
    let true_long = l0 + c;
    let omega = 125.04 - 1934.136 * jc;
    let lambda = true_long - 0.00569 - 0.00478 * omega.to_radians().sin();

    // Obliquity of the ecliptic
    let mean_obliq = 23.0
        + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0;
    let obliq_corr = mean_obliq + 0.00256 * omega.to_radians().cos();

    let lambda_rad = lambda.to_radians();
    let obliq_rad = obliq_corr.to_radians();

    // Solar declination
    let sin_dec = obliq_rad.sin() * lambda_rad.sin();
    let dec_rad = sin_dec.asin();

    // Equation of time (minutes)
    let y = (obliq_rad / 2.0).tan().powi(2);
    let l0_rad = l0.to_radians();
    let eq_time_min = 4.0
        * (y * (2.0 * l0_rad).sin() - 2.0 * e * m_rad.sin()
            + 4.0 * e * y * m_rad.sin() * (2.0 * l0_rad).cos()
            - 0.5 * y * y * (4.0 * l0_rad).sin()
            - 1.25 * e * e * (2.0 * m_rad).sin())
        .to_degrees();

    // True solar time
    let utc_minutes =
        dt.hour() as f64 * 60.0 + dt.minute() as f64 + dt.second() as f64 / 60.0;
    let true_solar_time_min = (utc_minutes + eq_time_min + 4.0 * lon_deg).rem_euclid(1440.0);

    // Hour angle (degrees)
    let hour_angle_deg = if true_solar_time_min / 4.0 < 0.0 {
        true_solar_time_min / 4.0 + 180.0
    } else {
        true_solar_time_min / 4.0 - 180.0
    };
    let ha_rad = hour_angle_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    // Solar zenith angle
    let cos_zenith = (lat_rad.sin() * dec_rad.sin()
        + lat_rad.cos() * dec_rad.cos() * ha_rad.cos())
    .clamp(-1.0, 1.0);
    let zenith_rad = cos_zenith.acos();
    let zenith_deg = zenith_rad.to_degrees();
    let geom_elev_deg = 90.0 - zenith_deg;

    // Solar azimuth
    let az_denom = lat_rad.cos() * zenith_rad.sin();
    let azimuth_deg = if az_denom.abs() > 1e-9 {
        let cos_az = ((lat_rad.sin() * cos_zenith - dec_rad.sin()) / az_denom).clamp(-1.0, 1.0);
        let base_az = cos_az.acos().to_degrees();
        if hour_angle_deg > 0.0 {
            (base_az + 180.0).rem_euclid(360.0)
        } else {
            (540.0 - base_az).rem_euclid(360.0)
        }
    } else if lat_deg > 0.0 {
        180.0
    } else {
        0.0
    };

    // Atmospheric refraction correction (Bennett / NOAA formula)
    let refraction_deg = atmospheric_refraction_deg(geom_elev_deg);
    let apparent_elev_deg = geom_elev_deg + refraction_deg;

    SolarPosition {
        azimuth_deg,
        geometric_elevation_deg: geom_elev_deg,
        apparent_elevation_deg: apparent_elev_deg,
        is_golden_hour: (-1.0..=6.5).contains(&apparent_elev_deg),
    }
}

/// Standard atmospheric refraction in degrees as a function of geometric elevation.
pub fn atmospheric_refraction_deg(geom_elev_deg: f64) -> f64 {
    if geom_elev_deg > 85.0 {
        0.0
    } else {
        let te = geom_elev_deg.to_radians().tan();
        let r_arcsec = if geom_elev_deg > 5.0 {
            58.1 / te - 0.07 / te.powi(3) + 0.000086 / te.powi(5)
        } else if geom_elev_deg > -0.575 {
            1735.0
                + geom_elev_deg
                    * (-518.2
                        + geom_elev_deg
                            * (103.4 + geom_elev_deg * (-12.79 + geom_elev_deg * 0.711)))
        } else {
            -20.774 / te
        };
        r_arcsec / 3600.0
    }
}

fn julian_day(dt: DateTime<Utc>) -> f64 {
    let mut y = dt.year() as i64;
    let mut m = dt.month() as i64;
    let d = dt.day() as f64
        + (dt.hour() as f64 + dt.minute() as f64 / 60.0 + dt.second() as f64 / 3600.0) / 24.0;

    if m <= 2 {
        y -= 1;
        m += 12;
    }
    let a = y / 100;
    let b = 2 - a + (a / 4);
    (365.25 * (y + 4716) as f64).floor()
        + (30.6001 * (m + 1) as f64).floor()
        + d
        + b as f64
        - 1524.5
}

/// Returns the annual (sunrise_min_az, sunrise_max_az, sunset_min_az, sunset_max_az)
/// for a given latitude, expanded by 15 degrees to account for high terrain horizons.
pub fn annual_solar_azimuth_windows(lat_deg: f64) -> ((f64, f64), (f64, f64)) {
    let lat_rad = lat_deg.clamp(-65.0, 65.0).to_radians();
    let max_dec_rad = 23.44_f64.to_radians();
    let cos_az = (max_dec_rad.sin() / lat_rad.cos()).clamp(-0.99, 0.99);
    let solstice_offset = (90.0 - cos_az.acos().to_degrees()).abs() + 18.0; // +18° margin for high mountain horizons
    let sr_min = (90.0 - solstice_offset).max(0.0);
    let sr_max = (90.0 + solstice_offset).min(180.0);
    let ss_min = (270.0 - solstice_offset).max(180.0);
    let ss_max = (270.0 + solstice_offset).min(360.0);
    ((sr_min, sr_max), (ss_min, ss_max))
}

/// Searches across a given year (or range of days) to find exact timestamps when the sun
/// passes through a terrain Light Notch defined by `(notch_az_deg, notch_width_deg, floor_elev_deg, wall_elev_deg)`.
pub fn find_notch_solar_alignments(
    lat_deg: f64,
    lon_deg: f64,
    year: i32,
    notch_az_deg: f64,
    notch_width_deg: f64,
    floor_elev_deg: f64,
    wall_elev_deg: f64,
    max_results: usize,
) -> Vec<SolarAlignmentWindow> {
    let mut alignments = Vec::new();
    let half_w = (notch_width_deg / 2.0).max(0.25);
    let is_sunrise = notch_az_deg < 180.0;

    // Approximate UTC offset in hours from longitude for search window centering
    let approx_tz_hours = (lon_deg / 15.0).round() as i64;

    // Step every 2 days to find candidate days, then refine at 1-minute resolution
    for day_of_year in 1..=365 {
        let Some(date) = NaiveDate::from_yo_opt(year, day_of_year) else {
            continue;
        };

        // Search window in local solar hours: 04:30 to 10:30 for sunrise, 15:00 to 21:00 for sunset
        let (start_local_hr, end_local_hr) = if is_sunrise { (4, 11) } else { (15, 21) };

        let base_dt = Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap());
        let search_start =
            base_dt + chrono::Duration::hours(start_local_hr - approx_tz_hours);
        let search_end = base_dt + chrono::Duration::hours(end_local_hr - approx_tz_hours);

        // Coarse check every 6 minutes first
        let mut hit_coarse = false;
        let mut t = search_start;
        while t <= search_end {
            let pos = solar_position(lat_deg, lon_deg, t);
            let az_diff = (pos.azimuth_deg - notch_az_deg + 540.0).rem_euclid(360.0) - 180.0;
            if az_diff.abs() <= half_w + 1.5
                && pos.apparent_elevation_deg >= floor_elev_deg - 1.5
                && pos.apparent_elevation_deg <= wall_elev_deg + 1.5
            {
                hit_coarse = true;
                break;
            }
            t += chrono::Duration::minutes(6);
        }

        if !hit_coarse {
            continue;
        }

        // Fine 1-minute sweep
        let mut window_start: Option<DateTime<Utc>> = None;
        let mut window_end: Option<DateTime<Utc>> = None;
        let mut best_sample: Option<(DateTime<Utc>, SolarPosition, f64)> = None;

        let mut t = search_start;
        while t <= search_end {
            let pos = solar_position(lat_deg, lon_deg, t);
            let az_diff = (pos.azimuth_deg - notch_az_deg + 540.0).rem_euclid(360.0) - 180.0;

            // Sun upper limb clears notch floor, and sun center is still below flanking notch walls
            let clears_floor =
                pos.apparent_elevation_deg + SOLAR_DISC_RADIUS_DEG >= floor_elev_deg;
            let below_walls = pos.apparent_elevation_deg <= wall_elev_deg;

            if az_diff.abs() <= half_w && clears_floor && below_walls {
                if window_start.is_none() {
                    window_start = Some(t);
                }
                window_end = Some(t);

                let score = (half_w - az_diff.abs())
                    + (wall_elev_deg - pos.apparent_elevation_deg).min(2.0);
                if best_sample.as_ref().map_or(true, |b| score > b.2) {
                    best_sample = Some((t, pos, score));
                }
            }
            t += chrono::Duration::minutes(1);
        }

        if let (Some(ws), Some(we), Some((peak_t, peak_pos, _))) =
            (window_start, window_end, best_sample)
        {
            let local_t = peak_t + chrono::Duration::hours(approx_tz_hours);
            alignments.push(SolarAlignmentWindow {
                date: date.format("%Y-%m-%d").to_string(),
                start_utc: ws,
                peak_utc: peak_t,
                end_utc: we,
                peak_local_solar_time: local_t.format("%H:%M").to_string(),
                sun_azimuth_deg: (peak_pos.azimuth_deg * 100.0).round() / 100.0,
                sun_elevation_deg: (peak_pos.apparent_elevation_deg * 100.0).round() / 100.0,
                clearance_above_floor_deg: ((peak_pos.apparent_elevation_deg - floor_elev_deg)
                    * 100.0)
                    .round()
                    / 100.0,
                depth_below_walls_deg: ((wall_elev_deg - peak_pos.apparent_elevation_deg) * 100.0)
                    .round()
                    / 100.0,
                phase: if is_sunrise {
                    "Sunrise".to_string()
                } else {
                    "Sunset".to_string()
                },
            });

            if alignments.len() >= max_results {
                break;
            }
        }
    }

    alignments
}

/// Computes high-precision lunar azimuth, topocentric elevation, illuminated fraction,
/// and lunar phase name for a given WGS84 coordinate and UTC timestamp.
/// Uses Jean Meeus (*Astronomical Algorithms*, Ch. 47 & 48) periodic series + topocentric parallax.
pub fn lunar_position(lat_deg: f64, lon_deg: f64, dt: DateTime<Utc>) -> LunarPosition {
    let jd = julian_day(dt);
    let t = (jd - 2451545.0) / 36525.0; // Julian centuries since J2000.0

    // Moon's mean longitude L' (deg)
    let l_prime = (218.3164477
        + 481267.88123421 * t
        - 0.0015786 * t * t
        + t * t * t / 538841.0)
        .rem_euclid(360.0);

    // Mean elongation of the Moon D (deg)
    let d = (297.8501921
        + 445267.1114034 * t
        - 0.0018819 * t * t
        + t * t * t / 545868.0)
        .rem_euclid(360.0);

    // Sun's mean anomaly M (deg)
    let m = (357.5291092
        + 35999.0502909 * t
        - 0.0001536 * t * t
        + t * t * t / 24490000.0)
        .rem_euclid(360.0);

    // Moon's mean anomaly M' (deg)
    let m_prime = (134.9633964
        + 477198.8675055 * t
        + 0.0087414 * t * t
        + t * t * t / 69699.0)
        .rem_euclid(360.0);

    // Moon's argument of latitude F (deg)
    let f = (93.2720950
        + 483202.0175233 * t
        - 0.0036539 * t * t
        - t * t * t / 3526000.0)
        .rem_euclid(360.0);

    let d_rad = d.to_radians();
    let m_rad = m.to_radians();
    let mp_rad = m_prime.to_radians();
    let f_rad = f.to_radians();

    // Principal periodic terms for lunar ecliptic longitude (deg)
    let delta_lon = 6.288774 * mp_rad.sin()
        + 1.274027 * (2.0 * d_rad - mp_rad).sin()
        + 0.658314 * (2.0 * d_rad).sin()
        + 0.213618 * (2.0 * mp_rad).sin()
        - 0.185116 * m_rad.sin()
        - 0.114332 * (2.0 * f_rad).sin()
        + 0.058793 * (2.0 * d_rad - 2.0 * mp_rad).sin()
        + 0.057066 * (2.0 * d_rad - m_rad - mp_rad).sin()
        + 0.053322 * (2.0 * d_rad + mp_rad).sin()
        + 0.045758 * (2.0 * d_rad - m_rad).sin()
        - 0.040923 * (m_rad - mp_rad).sin()
        - 0.034720 * d_rad.sin()
        - 0.030383 * (m_rad + mp_rad).sin()
        + 0.015327 * (2.0 * d_rad - 2.0 * f_rad).sin();

    // Principal periodic terms for lunar ecliptic latitude (deg)
    let delta_lat = 5.128122 * f_rad.sin()
        + 0.280602 * (mp_rad + f_rad).sin()
        + 0.277693 * (mp_rad - f_rad).sin()
        + 0.173237 * (2.0 * d_rad - f_rad).sin()
        + 0.055413 * (2.0 * d_rad - mp_rad + f_rad).sin()
        + 0.046271 * (2.0 * d_rad - mp_rad - f_rad).sin()
        + 0.032573 * (2.0 * d_rad + f_rad).sin()
        + 0.017198 * (2.0 * mp_rad + f_rad).sin();

    // Principal periodic terms for Earth-Moon distance (km)
    let distance_km = 385000.56
        - 20905.355 * mp_rad.cos()
        - 3699.111 * (2.0 * d_rad - mp_rad).cos()
        - 2955.968 * (2.0 * d_rad).cos()
        - 569.925 * (2.0 * mp_rad).cos()
        + 246.158 * (2.0 * d_rad - 2.0 * mp_rad).cos()
        - 152.138 * (2.0 * d_rad - m_rad - mp_rad).cos()
        - 170.733 * (2.0 * d_rad + mp_rad).cos()
        - 204.586 * (2.0 * d_rad - m_rad).cos();

    let lambda_moon = (l_prime + delta_lon).rem_euclid(360.0);
    let beta_moon = delta_lat;

    // Obliquity of the ecliptic
    let eps_deg = 23.4392911 - 0.0130042 * t;
    let eps_rad = eps_deg.to_radians();
    let lam_rad = lambda_moon.to_radians();
    let bet_rad = beta_moon.to_radians();

    // Ecliptic -> Geocentric Equatorial (RA alpha, Dec delta)
    let sin_dec = bet_rad.sin() * eps_rad.cos() + bet_rad.cos() * eps_rad.sin() * lam_rad.sin();
    let dec_rad = sin_dec.clamp(-1.0, 1.0).asin();
    let y = lam_rad.sin() * eps_rad.cos() - bet_rad.tan() * eps_rad.sin();
    let x = lam_rad.cos();
    let ra_deg = y.atan2(x).to_degrees().rem_euclid(360.0);

    // Greenwich Mean Sidereal Time (GMST) & Local Sidereal Time (LST) in degrees
    let gmst_deg = (280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t)
        .rem_euclid(360.0);
    let lst_deg = (gmst_deg + lon_deg).rem_euclid(360.0);

    // Local Hour Angle H
    let ha_deg = (lst_deg - ra_deg + 540.0).rem_euclid(360.0) - 180.0;
    let ha_rad = ha_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    // Geocentric elevation & azimuth
    let sin_alt_geo =
        (lat_rad.sin() * dec_rad.sin() + lat_rad.cos() * dec_rad.cos() * ha_rad.cos()).clamp(-1.0, 1.0);
    let alt_geo_rad = sin_alt_geo.asin();
    let zenith_rad = std::f64::consts::FRAC_PI_2 - alt_geo_rad;

    let az_denom = lat_rad.cos() * zenith_rad.sin();
    let azimuth_deg = if az_denom.abs() > 1e-9 {
        let cos_az = ((lat_rad.sin() * sin_alt_geo - dec_rad.sin()) / az_denom).clamp(-1.0, 1.0);
        let base_az = cos_az.acos().to_degrees();
        if ha_deg > 0.0 {
            (base_az + 180.0).rem_euclid(360.0)
        } else {
            (540.0 - base_az).rem_euclid(360.0)
        }
    } else if lat_deg > 0.0 {
        180.0
    } else {
        0.0
    };

    // Topocentric parallax correction in altitude: pi = asin(R_earth / distance_km)
    let horiz_parallax_deg = (6378.14 / distance_km).clamp(-1.0, 1.0).asin().to_degrees();
    let parallax_in_alt_deg = horiz_parallax_deg * alt_geo_rad.cos();
    let geom_elev_deg = alt_geo_rad.to_degrees() - parallax_in_alt_deg;
    let apparent_elev_deg = geom_elev_deg + atmospheric_refraction_deg(geom_elev_deg);

    // Solar ecliptic longitude for accurate lunar phase elongation (0..360°)
    let l0_sun = (280.46646 + t * 36000.76983).rem_euclid(360.0);
    let c_sun = m_rad.sin() * 1.914602 + (2.0 * m_rad).sin() * 0.019993;
    let lambda_sun = (l0_sun + c_sun).rem_euclid(360.0);
    let phase_angle_deg = (lambda_moon - lambda_sun).rem_euclid(360.0);

    // Geocentric elongation psi between Moon and Sun
    let cos_psi = bet_rad.cos() * (lambda_moon - lambda_sun).to_radians().cos();
    let sin_psi = (1.0 - cos_psi * cos_psi).max(0.0).sqrt();
    // Phase angle i in radians (Meeus eq. 48.2)
    let i_rad = (149_597_870.0 * sin_psi)
        .atan2(distance_km - 149_597_870.0 * cos_psi);
    let illuminated_fraction = ((1.0 + i_rad.cos()) * 0.5).clamp(0.0, 1.0);

    let phase_name = match phase_angle_deg {
        a if !(12.0..348.0).contains(&a) => "New Moon",
        a if a < 78.0 => "Waxing Crescent",
        a if a < 102.0 => "First Quarter",
        a if a < 168.0 => "Waxing Gibbous",
        a if a < 192.0 => "Full Moon",
        a if a < 258.0 => "Waning Gibbous",
        a if a < 282.0 => "Third Quarter",
        _ => "Waning Crescent",
    }
    .to_string();

    LunarPosition {
        azimuth_deg: (azimuth_deg * 100.0).round() / 100.0,
        geometric_elevation_deg: (geom_elev_deg * 100.0).round() / 100.0,
        apparent_elevation_deg: (apparent_elev_deg * 100.0).round() / 100.0,
        illuminated_fraction: (illuminated_fraction * 1000.0).round() / 1000.0,
        phase_angle_deg: (phase_angle_deg * 10.0).round() / 10.0,
        phase_name,
        distance_km: distance_km.round(),
    }
}

/// Computes the horizontal azimuth and elevation of the Galactic Center (Sagittarius A*:
/// J2000 RA = 17h 45m 40.04s = 266.4168°, Dec = -29° 00' 28.1" = -29.0078°) for a given
/// WGS84 coordinate and UTC timestamp.
pub fn galactic_core_position(lat_deg: f64, lon_deg: f64, dt: DateTime<Utc>) -> GalacticCorePosition {
    let jd = julian_day(dt);
    let t = (jd - 2451545.0) / 36525.0;

    // J2000 coordinates of Sagittarius A* with first-order precession to epoch of date
    let ra0_deg = 266.4168_f64;
    let dec0_deg = -29.0078_f64;
    let ra0_rad = ra0_deg.to_radians();
    let dec0_rad = dec0_deg.to_radians();

    // Annual precession in RA (m_prec ≈ 0.012812°/yr, n_prec ≈ 0.005567°/yr)
    let years = t * 100.0;
    let d_ra_deg = (0.012812 + 0.005567 * ra0_rad.sin() * dec0_rad.tan()) * years;
    let d_dec_deg = 0.005567 * ra0_rad.cos() * years;

    let ra_deg = (ra0_deg + d_ra_deg).rem_euclid(360.0);
    let dec_rad = (dec0_deg + d_dec_deg).to_radians();

    let gmst_deg = (280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t)
        .rem_euclid(360.0);
    let lst_deg = (gmst_deg + lon_deg).rem_euclid(360.0);

    let ha_deg = (lst_deg - ra_deg + 540.0).rem_euclid(360.0) - 180.0;
    let ha_rad = ha_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    let sin_alt =
        (lat_rad.sin() * dec_rad.sin() + lat_rad.cos() * dec_rad.cos() * ha_rad.cos()).clamp(-1.0, 1.0);
    let geom_elev_deg = sin_alt.asin().to_degrees();
    let zenith_rad = (90.0 - geom_elev_deg).to_radians();

    let az_denom = lat_rad.cos() * zenith_rad.sin();
    let azimuth_deg = if az_denom.abs() > 1e-9 {
        let cos_az = ((lat_rad.sin() * sin_alt - dec_rad.sin()) / az_denom).clamp(-1.0, 1.0);
        let base_az = cos_az.acos().to_degrees();
        if ha_deg > 0.0 {
            (base_az + 180.0).rem_euclid(360.0)
        } else {
            (540.0 - base_az).rem_euclid(360.0)
        }
    } else if lat_deg > 0.0 {
        180.0
    } else {
        0.0
    };

    let apparent_elev_deg = geom_elev_deg + atmospheric_refraction_deg(geom_elev_deg);

    GalacticCorePosition {
        azimuth_deg: (azimuth_deg * 100.0).round() / 100.0,
        geometric_elevation_deg: (geom_elev_deg * 100.0).round() / 100.0,
        apparent_elevation_deg: (apparent_elev_deg * 100.0).round() / 100.0,
        is_above_horizon: apparent_elev_deg > 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yosemite_equinox_sunset() {
        // Yosemite Valley: 37.7456 N, -119.5383 W (~8h behind UTC) on March 20, 2026 near sunset (01:55 UTC = 18:55 PDT)
        let dt = Utc.with_ymd_and_hms(2026, 3, 21, 1, 55, 0).unwrap();
        let pos = solar_position(37.7456, -119.5383, dt);
        assert!(
            pos.azimuth_deg > 265.0 && pos.azimuth_deg < 275.0,
            "Expected sunset near due west (270°), got {}",
            pos.azimuth_deg
        );
        assert!(
            pos.apparent_elevation_deg.abs() < 4.0,
            "Expected low elevation near sunset, got {}",
            pos.apparent_elevation_deg
        );
    }

    #[test]
    fn test_lunar_and_galactic_core_ephemeris() {
        // Summer Milky Way season in Yosemite (July 15, 2026 at 06:00 UTC = 22:00 PST / 23:00 PDT)
        let dt = Utc.with_ymd_and_hms(2026, 7, 15, 6, 0, 0).unwrap();
        let gc = galactic_core_position(37.7456, -119.5383, dt);
        assert!(
            gc.is_above_horizon && gc.apparent_elevation_deg > 15.0,
            "Expected Galactic Core well above southern horizon in July night, got el={}°, az={}°",
            gc.apparent_elevation_deg,
            gc.azimuth_deg
        );
        assert!(
            gc.azimuth_deg > 150.0 && gc.azimuth_deg < 210.0,
            "Expected Galactic Core in the south (150°..210°), got {}°",
            gc.azimuth_deg
        );

        let moon = lunar_position(37.7456, -119.5383, dt);
        assert!(
            (0.0..=1.0).contains(&moon.illuminated_fraction),
            "Invalid lunar illumination fraction: {}",
            moon.illuminated_fraction
        );
        assert!(
            (350_000.0..=410_000.0).contains(&moon.distance_km),
            "Unexpected lunar distance: {} km",
            moon.distance_km
        );
    }
}
