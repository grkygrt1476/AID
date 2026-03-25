/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#include "gstdsposepatchpersistent.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

GST_DEBUG_CATEGORY_STATIC (gst_dsposepatchpersistent_debug);
#define GST_CAT_DEFAULT gst_dsposepatchpersistent_debug

enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_PROCESS_FULL_FRAME,
  PROP_BATCH_SIZE,
  PROP_BLUR_OBJECTS,
  PROP_GPU_DEVICE_ID
};

#define DEFAULT_UNIQUE_ID 17
#define DEFAULT_PROCESSING_WIDTH 960
#define DEFAULT_PROCESSING_HEIGHT 540
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_BLUR_OBJECTS FALSE
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_HARD_MAX_PROXY_AGE_FRAMES 18
#define DEFAULT_MAX_CENTER_SHIFT_PX 32.0
#define DEFAULT_MIN_GOOD_POINTS 4
#define DEFAULT_FEATURE_MAX_CORNERS 64
#define DEFAULT_LK_WIN_SIZE 21
#define DEFAULT_PATCH_WIDTH_RATIO 0.52
#define DEFAULT_PATCH_HEIGHT_RATIO 0.36
#define DEFAULT_PATCH_Y_OFFSET_RATIO 0.22
#define DEFAULT_HOLD_AFTER_FAIL_FRAMES 3
#define DEFAULT_MIN_PATCH_SIZE_PX 20
#define DEFAULT_PERSON_CLASS_ID 0

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

static GstStaticPadTemplate gst_dsposepatchpersistent_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsposepatchpersistent_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

#define gst_dsposepatchpersistent_parent_class parent_class
G_DEFINE_TYPE (GstDsPosePatchPersistent, gst_dsposepatchpersistent, GST_TYPE_BASE_TRANSFORM);

static void gst_dsposepatchpersistent_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsposepatchpersistent_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean gst_dsposepatchpersistent_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dsposepatchpersistent_start (GstBaseTransform * btrans);
static gboolean gst_dsposepatchpersistent_stop (GstBaseTransform * btrans);
static GstFlowReturn gst_dsposepatchpersistent_transform_ip (GstBaseTransform * btrans,
    GstBuffer * inbuf);

static inline guint
read_env_uint (const char *name, guint default_value, guint min_value)
{
  const gchar *raw = g_getenv (name);
  if (!raw || !*raw)
    return default_value;

  gchar *end = NULL;
  guint64 parsed = g_ascii_strtoull (raw, &end, 10);
  if (!end || *end != '\0' || parsed < min_value || parsed > G_MAXUINT)
    return default_value;
  return (guint) parsed;
}

static inline gdouble
read_env_double (const char *name, gdouble default_value, gdouble min_value)
{
  const gchar *raw = g_getenv (name);
  if (!raw || !*raw)
    return default_value;

  gchar *end = NULL;
  gdouble parsed = g_ascii_strtod (raw, &end);
  if (!end || *end != '\0' || parsed < min_value)
    return default_value;
  return parsed;
}

static inline gint
read_env_int (const char *name, gint default_value, gint min_value)
{
  const gchar *raw = g_getenv (name);
  if (!raw || !*raw)
    return default_value;

  gchar *end = NULL;
  gint64 parsed = g_ascii_strtoll (raw, &end, 10);
  if (!end || *end != '\0' || parsed < min_value || parsed > G_MAXINT)
    return default_value;
  return (gint) parsed;
}

static inline gboolean
read_env_bool (const char *name, gboolean default_value)
{
  const gchar *raw = g_getenv (name);
  if (!raw || !*raw)
    return default_value;

  gchar *lower = g_ascii_strdown (raw, -1);
  gboolean value = default_value;
  if (g_strcmp0 (lower, "1") == 0 || g_strcmp0 (lower, "true") == 0 ||
      g_strcmp0 (lower, "yes") == 0 || g_strcmp0 (lower, "on") == 0) {
    value = TRUE;
  } else if (g_strcmp0 (lower, "0") == 0 || g_strcmp0 (lower, "false") == 0 ||
      g_strcmp0 (lower, "no") == 0 || g_strcmp0 (lower, "off") == 0) {
    value = FALSE;
  }
  g_free (lower);
  return value;
}

static inline std::string
make_track_key (guint source_id, guint64 object_id)
{
  return std::to_string (source_id) + ":" + std::to_string ((unsigned long long) object_id);
}

static inline bool
is_valid_rect (const cv::Rect2f &rect)
{
  return rect.width >= 1.0f && rect.height >= 1.0f;
}

static inline cv::Rect2f
clamp_rect (const cv::Rect2f &rect, gint width, gint height)
{
  float x = std::max (0.0f, rect.x);
  float y = std::max (0.0f, rect.y);
  float max_w = std::max (1.0f, (float) width - x);
  float max_h = std::max (1.0f, (float) height - y);
  float w = std::min (std::max (0.0f, rect.width), max_w);
  float h = std::min (std::max (0.0f, rect.height), max_h);
  return cv::Rect2f (x, y, w, h);
}

static inline cv::Point2f
rect_center (const cv::Rect2f &rect)
{
  return cv::Point2f (rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
}

static inline cv::Rect
to_roi (const cv::Rect2f &rect, gint width, gint height)
{
  cv::Rect2f bounded = clamp_rect (rect, width, height);
  gint x = std::max (0, (gint) std::floor (bounded.x));
  gint y = std::max (0, (gint) std::floor (bounded.y));
  gint w = std::max (1, (gint) std::round (bounded.width));
  gint h = std::max (1, (gint) std::round (bounded.height));
  if (x + w > width)
    w = std::max (1, width - x);
  if (y + h > height)
    h = std::max (1, height - y);
  return cv::Rect (x, y, w, h);
}

static inline cv::Rect2f
original_to_proc_rect (GstDsPosePatchPersistent *self, const NvOSD_RectParams &rect)
{
  const gdouble scale_x = (gdouble) self->processing_width / (gdouble) self->video_info.width;
  const gdouble scale_y = (gdouble) self->processing_height / (gdouble) self->video_info.height;
  return clamp_rect (
      cv::Rect2f ((float) (rect.left * scale_x), (float) (rect.top * scale_y),
          (float) (rect.width * scale_x), (float) (rect.height * scale_y)),
      self->processing_width, self->processing_height);
}

static inline cv::Rect2f
proc_to_original_rect (GstDsPosePatchPersistent *self, const cv::Rect2f &rect)
{
  const gdouble scale_x = (gdouble) self->video_info.width / (gdouble) self->processing_width;
  const gdouble scale_y = (gdouble) self->video_info.height / (gdouble) self->processing_height;
  return clamp_rect (
      cv::Rect2f ((float) (rect.x * scale_x), (float) (rect.y * scale_y),
          (float) (rect.width * scale_x), (float) (rect.height * scale_y)),
      self->video_info.width, self->video_info.height);
}

static inline cv::Point2f
proc_to_original_point (GstDsPosePatchPersistent *self, const cv::Point2f &point)
{
  const gdouble scale_x = (gdouble) self->video_info.width / (gdouble) self->processing_width;
  const gdouble scale_y = (gdouble) self->video_info.height / (gdouble) self->processing_height;
  return cv::Point2f ((float) (point.x * scale_x), (float) (point.y * scale_y));
}

static inline float
proc_shift_limit (GstDsPosePatchPersistent *self)
{
  return (float) (self->max_center_shift_px * ((gdouble) self->processing_width / (gdouble) self->video_info.width));
}

static float
median_in_place (std::vector<float> &values)
{
  if (values.empty ())
    return 0.0f;

  auto mid = values.begin () + values.size () / 2;
  std::nth_element (values.begin (), mid, values.end ());
  float median = *mid;
  if ((values.size () % 2) == 0) {
    auto lower = values.begin () + (values.size () / 2) - 1;
    std::nth_element (values.begin (), lower, values.end ());
    median = 0.5f * (median + *lower);
  }
  return median;
}

static bool
try_extract_pose_anchor_from_meta (GstDsPosePatchPersistent *self, NvDsObjectMeta *obj_meta,
    cv::Point2f &anchor_proc, std::string &source_label)
{
  (void) self;
  (void) obj_meta;
  (void) anchor_proc;
  source_label = "bbox_fallback";
  return false;
}

static bool
build_patch_geometry (GstDsPosePatchPersistent *self, NvDsObjectMeta *obj_meta,
    const cv::Rect2f &bbox_proc, PersistentPatchGeometry &patch,
    std::string &failure_reason)
{
  cv::Point2f anchor_proc;
  std::string source_label = "bbox_fallback";
  const bool has_pose_anchor =
      try_extract_pose_anchor_from_meta (self, obj_meta, anchor_proc, source_label);

  if (!has_pose_anchor) {
    anchor_proc.x = bbox_proc.x + bbox_proc.width * 0.5f;
    anchor_proc.y = bbox_proc.y + bbox_proc.height * (float) self->patch_y_offset_ratio;
    source_label = "bbox_fallback";
  }

  const float patch_width =
      std::max ((float) self->min_patch_size_px, bbox_proc.width * (float) self->patch_width_ratio);
  const float patch_height =
      std::max ((float) self->min_patch_size_px, bbox_proc.height * (float) self->patch_height_ratio);

  cv::Rect2f patch_rect (
      anchor_proc.x - patch_width * 0.5f,
      anchor_proc.y - patch_height * 0.5f,
      patch_width,
      patch_height);
  patch_rect = clamp_rect (patch_rect, self->processing_width, self->processing_height);

  if (patch_rect.width < (float) self->min_patch_size_px ||
      patch_rect.height < (float) self->min_patch_size_px) {
    failure_reason = "patch_too_small";
    return false;
  }

  patch.patch_rect_proc = patch_rect;
  patch.anchor_proc = anchor_proc;
  patch.source = source_label;
  return true;
}

static bool
init_features_for_patch (const cv::Mat &gray, const cv::Rect2f &patch_rect,
    guint feature_max_corners, std::vector<cv::Point2f> &features)
{
  if (gray.empty () || !is_valid_rect (patch_rect))
    return false;

  cv::Rect roi = to_roi (patch_rect, gray.cols, gray.rows);
  if (roi.width < 4 || roi.height < 4)
    return false;

  std::vector<cv::Point2f> local_points;
  cv::goodFeaturesToTrack (gray (roi), local_points, (int) feature_max_corners,
      0.003, 2.0, cv::Mat (), 3, false, 0.04);

  features.clear ();
  for (const cv::Point2f &point : local_points)
    features.emplace_back (point.x + roi.x, point.y + roi.y);
  return !features.empty ();
}

static bool
is_plausible_handoff (const cv::Rect2f &candidate, const cv::Rect2f &proxy,
    float max_shift_proc)
{
  const cv::Point2f candidate_center = rect_center (candidate);
  const cv::Point2f proxy_center = rect_center (proxy);
  const float dx = candidate_center.x - proxy_center.x;
  const float dy = candidate_center.y - proxy_center.y;
  const float center_dist = std::sqrt (dx * dx + dy * dy);
  if (center_dist > max_shift_proc)
    return false;

  const float width_ratio = candidate.width / std::max (1.0f, proxy.width);
  const float height_ratio = candidate.height / std::max (1.0f, proxy.height);
  return width_ratio >= 0.5f && width_ratio <= 2.0f &&
      height_ratio >= 0.5f && height_ratio <= 2.0f;
}

static void
write_sidecar_row (GstDsPosePatchPersistent *self, NvDsFrameMeta *frame_meta,
    guint64 track_id, const char *mode, gboolean proxy_active, guint proxy_age,
    const std::string &event_name, const std::string &stop_reason,
    const std::string &handoff_reason, const cv::Rect2f &bbox_proc,
    const cv::Rect2f &patch_proc, const std::string &patch_source,
    guint tracked_points, const cv::Point2f &flow_delta_proc, float flow_mag_proc)
{
  if (!self->sidecar_stream || !self->sidecar_stream->is_open ())
    return;

  cv::Rect2f bbox = proc_to_original_rect (self, bbox_proc);
  cv::Rect2f patch = proc_to_original_rect (self, patch_proc);
  (*self->sidecar_stream)
      << frame_meta->frame_num << ","
      << frame_meta->source_id << ","
      << (unsigned long long) track_id << ","
      << mode << ","
      << (proxy_active ? 1 : 0) << ","
      << proxy_age << ","
      << event_name << ","
      << stop_reason << ","
      << handoff_reason << ","
      << bbox.x << ","
      << bbox.y << ","
      << bbox.width << ","
      << bbox.height << ","
      << patch.x << ","
      << patch.y << ","
      << patch.width << ","
      << patch.height << ","
      << patch_source << ","
      << tracked_points << ","
      << flow_delta_proc.x << ","
      << flow_delta_proc.y << ","
      << flow_mag_proc << "\n";
  self->sidecar_stream->flush ();
}

static void
add_proxy_overlay (GstDsPosePatchPersistent *self, NvDsBatchMeta *batch_meta,
    NvDsFrameMeta *frame_meta, guint64 track_id, guint proxy_age,
    const char *mode, const cv::Rect2f &bbox_proc, const cv::Rect2f &patch_proc,
    guint tracked_points, const std::vector<cv::Point2f> &patch_points)
{
  const guint max_overlay_points = 12;
  NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
  if (!display_meta)
    return;

  cv::Rect2f bbox = proc_to_original_rect (self, bbox_proc);
  display_meta->num_rects = 1;
  NvOSD_RectParams &bbox_rect = display_meta->rect_params[0];
  bbox_rect.left = bbox.x;
  bbox_rect.top = bbox.y;
  bbox_rect.width = bbox.width;
  bbox_rect.height = bbox.height;
  bbox_rect.border_width = 3;
  bbox_rect.has_bg_color = 0;
  if (g_strcmp0 (mode, "frozen_hold") == 0) {
    bbox_rect.border_color.red = 1.0;
    bbox_rect.border_color.green = 1.0;
    bbox_rect.border_color.blue = 0.0;
  } else {
    bbox_rect.border_color.red = 1.0;
    bbox_rect.border_color.green = 0.55;
    bbox_rect.border_color.blue = 0.0;
  }
  bbox_rect.border_color.alpha = 1.0;

  if (is_valid_rect (patch_proc)) {
    cv::Rect2f patch = proc_to_original_rect (self, patch_proc);
    NvOSD_RectParams &patch_rect = display_meta->rect_params[display_meta->num_rects++];
    patch_rect.left = patch.x;
    patch_rect.top = patch.y;
    patch_rect.width = patch.width;
    patch_rect.height = patch.height;
    patch_rect.border_width = 2;
    patch_rect.has_bg_color = 0;
    if (g_strcmp0 (mode, "frozen_hold") == 0) {
      patch_rect.border_color.red = 1.0;
      patch_rect.border_color.green = 1.0;
      patch_rect.border_color.blue = 0.2;
    } else {
      patch_rect.border_color.red = 0.1;
      patch_rect.border_color.green = 1.0;
      patch_rect.border_color.blue = 1.0;
    }
    patch_rect.border_color.alpha = 1.0;
  }

  cv::Point2f anchor_proc = is_valid_rect (patch_proc) ? rect_center (patch_proc) : rect_center (bbox_proc);
  cv::Point2f anchor = proc_to_original_point (self, anchor_proc);
  NvOSD_RectParams &anchor_rect = display_meta->rect_params[display_meta->num_rects++];
  anchor_rect.left = std::max (0.0f, anchor.x - 3.0f);
  anchor_rect.top = std::max (0.0f, anchor.y - 3.0f);
  anchor_rect.width = 6.0f;
  anchor_rect.height = 6.0f;
  anchor_rect.border_width = 2;
  anchor_rect.has_bg_color = 1;
  anchor_rect.border_color.red = 1.0;
  anchor_rect.border_color.green = 0.2;
  anchor_rect.border_color.blue = 0.2;
  anchor_rect.border_color.alpha = 1.0;
  anchor_rect.bg_color = (NvOSD_ColorParams) {1.0, 0.2, 0.2, 0.7};

  if (!patch_points.empty ()) {
    const guint point_count = (guint) patch_points.size ();
    const guint stride = std::max (1u, (point_count + max_overlay_points - 1) / max_overlay_points);
    for (guint idx = 0, drawn = 0; idx < point_count && drawn < max_overlay_points; idx += stride, ++drawn) {
      cv::Point2f point = proc_to_original_point (self, patch_points[idx]);
      NvOSD_RectParams &point_rect = display_meta->rect_params[display_meta->num_rects++];
      point_rect.left = std::max (0.0f, point.x - 2.0f);
      point_rect.top = std::max (0.0f, point.y - 2.0f);
      point_rect.width = 4.0f;
      point_rect.height = 4.0f;
      point_rect.border_width = 1;
      point_rect.has_bg_color = 1;
      point_rect.border_color.red = 0.1;
      point_rect.border_color.green = 1.0;
      point_rect.border_color.blue = 1.0;
      point_rect.border_color.alpha = 1.0;
      point_rect.bg_color = (NvOSD_ColorParams) {0.1, 1.0, 1.0, 0.8};
    }
  }

  display_meta->num_labels = 1;
  NvOSD_TextParams &text = display_meta->text_params[0];
  if (g_strcmp0 (mode, "frozen_hold") == 0) {
    text.display_text = g_strdup_printf ("PPP %" G_GUINT64_FORMAT " hold +%u pts=%u",
        (guint64) track_id, proxy_age, tracked_points);
  } else {
    text.display_text = g_strdup_printf ("PPP %" G_GUINT64_FORMAT " +%u pts=%u",
        (guint64) track_id, proxy_age, tracked_points);
  }
  text.x_offset = (gint) bbox.x;
  text.y_offset = std::max (0, (gint) bbox.y - 12);
  text.set_bg_clr = 1;
  text.text_bg_clr = (NvOSD_ColorParams) {0.0, 0.0, 0.0, 0.8};
  text.font_params.font_name = (gchar *) "Serif";
  text.font_params.font_size = 12;
  text.font_params.font_color = (NvOSD_ColorParams) {1.0, 0.75, 0.2, 1.0};

  nvds_add_display_meta_to_frame (frame_meta, display_meta);
}

static GstFlowReturn
convert_frame_to_gray (GstDsPosePatchPersistent *self, NvBufSurface *input_buf,
    guint batch_id, cv::Mat &out_gray)
{
  NvBufSurface ip_surf = *input_buf;
  ip_surf.numFilled = ip_surf.batchSize = 1;
  ip_surf.surfaceList = &(input_buf->surfaceList[batch_id]);

  NvBufSurfTransformConfigParams session_params = self->transform_config_params;
  session_params.cuda_stream = self->cuda_stream;
  if (NvBufSurfTransformSetSessionParams (&session_params) != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (self, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed"), (NULL));
    return GST_FLOW_ERROR;
  }

  NvBufSurfTransformRect src_rect = {
    0, 0,
    input_buf->surfaceList[batch_id].width,
    input_buf->surfaceList[batch_id].height
  };
  NvBufSurfTransformRect dst_rect = {
    0, 0,
    (guint) self->processing_width,
    (guint) self->processing_height
  };

  NvBufSurfTransformParams transform_params;
  std::memset (&transform_params, 0, sizeof (transform_params));
  transform_params.src_rect = &src_rect;
  transform_params.dst_rect = &dst_rect;
  transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  transform_params.transform_filter = NvBufSurfTransformInter_Default;

  NvBufSurfaceMemSet (self->inter_buf, 0, 0, 0);
  if (NvBufSurfTransform (&ip_surf, self->inter_buf, &transform_params) !=
      NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (self, STREAM, FAILED,
        ("NvBufSurfTransform failed while converting frame"), (NULL));
    return GST_FLOW_ERROR;
  }

  if (NvBufSurfaceMap (self->inter_buf, 0, 0, NVBUF_MAP_READ) != 0) {
    GST_ELEMENT_ERROR (self, STREAM, FAILED,
        ("NvBufSurfaceMap failed for assist buffer"), (NULL));
    return GST_FLOW_ERROR;
  }

  NvBufSurfaceSyncForCpu (self->inter_buf, 0, 0);
  cv::Mat rgba (self->processing_height, self->processing_width, CV_8UC4,
      self->inter_buf->surfaceList[0].mappedAddr.addr[0],
      self->inter_buf->surfaceList[0].pitch);
  cv::cvtColor (rgba, out_gray, cv::COLOR_RGBA2GRAY);
  NvBufSurfaceUnMap (self->inter_buf, 0, 0);
  return GST_FLOW_OK;
}

static bool
advance_patch_proxy (GstDsPosePatchPersistent *self, const cv::Mat &prev_gray,
    const cv::Mat &current_gray, PersistentTrackState &state,
    cv::Rect2f &out_bbox_proc, cv::Rect2f &out_patch_proc,
    std::vector<cv::Point2f> &out_points, guint &out_good_points,
    cv::Point2f &out_flow_delta_proc, float &out_flow_mag_proc,
    std::string &failure_reason)
{
  if (!is_valid_rect (state.last_patch_rect_proc)) {
    failure_reason = "patch_unavailable";
    return false;
  }

  std::vector<cv::Point2f> seed_points = state.patch_points;
  if (seed_points.size () < self->min_good_points &&
      !init_features_for_patch (prev_gray, state.last_patch_rect_proc,
          self->feature_max_corners, seed_points)) {
    failure_reason = "too_little_texture";
    return false;
  }

  std::vector<cv::Point2f> next_points;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK (prev_gray, current_gray, seed_points, next_points,
      status, err, cv::Size ((int) self->lk_win_size, (int) self->lk_win_size),
      3, cv::TermCriteria (cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

  std::vector<cv::Point2f> prev_good;
  std::vector<cv::Point2f> next_good;
  for (guint i = 0; i < status.size (); ++i) {
    if (!status[i])
      continue;
    const cv::Point2f &pt = next_points[i];
    if (pt.x < 0.0f || pt.y < 0.0f || pt.x >= current_gray.cols || pt.y >= current_gray.rows)
      continue;
    prev_good.push_back (seed_points[i]);
    next_good.push_back (pt);
  }

  out_good_points = (guint) next_good.size ();
  if (out_good_points < self->min_good_points) {
    failure_reason = "too_few_good_points";
    return false;
  }

  std::vector<float> dx_values;
  std::vector<float> dy_values;
  std::vector<float> mag_values;
  dx_values.reserve (out_good_points);
  dy_values.reserve (out_good_points);
  mag_values.reserve (out_good_points);
  for (guint i = 0; i < out_good_points; ++i) {
    const float dx = next_good[i].x - prev_good[i].x;
    const float dy = next_good[i].y - prev_good[i].y;
    dx_values.push_back (dx);
    dy_values.push_back (dy);
    mag_values.push_back (std::sqrt (dx * dx + dy * dy));
  }

  const float dx = median_in_place (dx_values);
  const float dy = median_in_place (dy_values);
  const float shift_mag = median_in_place (mag_values);
  if (shift_mag > proc_shift_limit (self)) {
    failure_reason = "shift_too_large";
    return false;
  }

  out_flow_delta_proc = cv::Point2f (dx, dy);
  out_flow_mag_proc = shift_mag;
  out_patch_proc = clamp_rect (
      cv::Rect2f (state.last_patch_rect_proc.x + dx, state.last_patch_rect_proc.y + dy,
          state.last_patch_rect_proc.width, state.last_patch_rect_proc.height),
      current_gray.cols, current_gray.rows);
  out_bbox_proc = clamp_rect (
      cv::Rect2f (state.last_bbox_proc.x + dx, state.last_bbox_proc.y + dy,
          state.last_bbox_proc.width, state.last_bbox_proc.height),
      current_gray.cols, current_gray.rows);
  out_points = next_good;
  return true;
}

static void
gst_dsposepatchpersistent_class_init (GstDsPosePatchPersistentClass *klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstElementClass *gstelement_class = (GstElementClass *) klass;
  GstBaseTransformClass *gstbasetransform_class = (GstBaseTransformClass *) klass;

  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsposepatchpersistent_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsposepatchpersistent_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsposepatchpersistent_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsposepatchpersistent_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsposepatchpersistent_stop);
  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_dsposepatchpersistent_transform_ip);

  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width", "Processing Width",
          "Width of the frame used for persistent pose-patch assist", 1, G_MAXINT,
          DEFAULT_PROCESSING_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height", "Processing Height",
          "Height of the frame used for persistent pose-patch assist", 1, G_MAXINT,
          DEFAULT_PROCESSING_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_FULL_FRAME,
      g_param_spec_boolean ("full-frame", "Full Frame",
          "Required for the persistent pose-patch assist path", DEFAULT_PROCESS_FULL_FRAME,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BLUR_OBJECTS,
      g_param_spec_boolean ("blur-objects", "Blur Objects",
          "Unused compatibility property", DEFAULT_BLUR_OBJECTS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID", 0, G_MAXUINT, DEFAULT_GPU_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for processing", 1, NVDSEXAMPLE_MAX_BATCH_SIZE,
          DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsposepatchpersistent_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsposepatchpersistent_sink_template));

  gst_element_class_set_details_simple (gstelement_class,
      "DsPosePatchPersistent plugin",
      "DsPosePatchPersistent Plugin",
      "Persistent upper-body patch continuity debug assist for DeepStream tracker misses",
      "AID");
}

static void
gst_dsposepatchpersistent_init (GstDsPosePatchPersistent *self)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (self);
  gst_base_transform_set_in_place (btrans, TRUE);
  gst_base_transform_set_passthrough (btrans, TRUE);

  self->unique_id = DEFAULT_UNIQUE_ID;
  self->processing_width = DEFAULT_PROCESSING_WIDTH;
  self->processing_height = DEFAULT_PROCESSING_HEIGHT;
  self->process_full_frame = DEFAULT_PROCESS_FULL_FRAME;
  self->blur_objects = DEFAULT_BLUR_OBJECTS;
  self->gpu_id = DEFAULT_GPU_ID;
  self->max_batch_size = DEFAULT_BATCH_SIZE;

  self->hard_max_proxy_age_frames = DEFAULT_HARD_MAX_PROXY_AGE_FRAMES;
  self->max_center_shift_px = DEFAULT_MAX_CENTER_SHIFT_PX;
  self->min_good_points = DEFAULT_MIN_GOOD_POINTS;
  self->feature_max_corners = DEFAULT_FEATURE_MAX_CORNERS;
  self->lk_win_size = DEFAULT_LK_WIN_SIZE;
  self->patch_width_ratio = DEFAULT_PATCH_WIDTH_RATIO;
  self->patch_height_ratio = DEFAULT_PATCH_HEIGHT_RATIO;
  self->patch_y_offset_ratio = DEFAULT_PATCH_Y_OFFSET_RATIO;
  self->freeze_on_patch_fail = TRUE;
  self->hold_after_fail_frames = DEFAULT_HOLD_AFTER_FAIL_FRAMES;
  self->min_patch_size_px = DEFAULT_MIN_PATCH_SIZE_PX;
  self->person_class_id = DEFAULT_PERSON_CLASS_ID;

  self->track_states = NULL;
  self->prev_gray = NULL;
  self->inter_buf = NULL;
  self->cuda_stream = NULL;
  self->sidecar_path = NULL;
  self->sidecar_stream = NULL;
}

static void
gst_dsposepatchpersistent_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsPosePatchPersistent *self = GST_DSPOSEPATCHPERSISTENT (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      self->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      self->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      self->processing_height = g_value_get_int (value);
      break;
    case PROP_PROCESS_FULL_FRAME:
      self->process_full_frame = g_value_get_boolean (value);
      break;
    case PROP_BLUR_OBJECTS:
      self->blur_objects = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      self->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BATCH_SIZE:
      self->max_batch_size = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_dsposepatchpersistent_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsPosePatchPersistent *self = GST_DSPOSEPATCHPERSISTENT (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, self->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, self->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, self->processing_height);
      break;
    case PROP_PROCESS_FULL_FRAME:
      g_value_set_boolean (value, self->process_full_frame);
      break;
    case PROP_BLUR_OBJECTS:
      g_value_set_boolean (value, self->blur_objects);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, self->gpu_id);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, self->max_batch_size);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_dsposepatchpersistent_start (GstBaseTransform * btrans)
{
  GstDsPosePatchPersistent *self = GST_DSPOSEPATCHPERSISTENT (btrans);

  self->hard_max_proxy_age_frames =
      read_env_uint ("AID_DSPOSEPATCHPERSISTENT_HARD_MAX_PROXY_AGE_FRAMES",
          DEFAULT_HARD_MAX_PROXY_AGE_FRAMES, 1);
  self->max_center_shift_px =
      read_env_double ("AID_DSPOSEPATCHPERSISTENT_MAX_CENTER_SHIFT_PX",
          DEFAULT_MAX_CENTER_SHIFT_PX, 1.0);
  self->min_good_points =
      read_env_uint ("AID_DSPOSEPATCHPERSISTENT_MIN_GOOD_POINTS",
          DEFAULT_MIN_GOOD_POINTS, 1);
  self->feature_max_corners =
      read_env_uint ("AID_DSPOSEPATCHPERSISTENT_FEATURE_MAX_CORNERS",
          DEFAULT_FEATURE_MAX_CORNERS, 1);
  self->lk_win_size =
      read_env_uint ("AID_DSPOSEPATCHPERSISTENT_LK_WIN_SIZE",
          DEFAULT_LK_WIN_SIZE, 3);
  self->patch_width_ratio =
      read_env_double ("AID_DSPOSEPATCHPERSISTENT_PATCH_WIDTH_RATIO",
          DEFAULT_PATCH_WIDTH_RATIO, 0.05);
  self->patch_height_ratio =
      read_env_double ("AID_DSPOSEPATCHPERSISTENT_PATCH_HEIGHT_RATIO",
          DEFAULT_PATCH_HEIGHT_RATIO, 0.05);
  self->patch_y_offset_ratio =
      read_env_double ("AID_DSPOSEPATCHPERSISTENT_PATCH_Y_OFFSET_RATIO",
          DEFAULT_PATCH_Y_OFFSET_RATIO, 0.0);
  self->freeze_on_patch_fail =
      read_env_bool ("AID_DSPOSEPATCHPERSISTENT_FREEZE_ON_PATCH_FAIL", TRUE);
  self->hold_after_fail_frames =
      read_env_uint ("AID_DSPOSEPATCHPERSISTENT_HOLD_AFTER_FAIL_FRAMES",
          DEFAULT_HOLD_AFTER_FAIL_FRAMES, 0);
  self->person_class_id =
      read_env_int ("AID_DSPOSEPATCHPERSISTENT_PERSON_CLASS_ID",
          DEFAULT_PERSON_CLASS_ID, 0);

  const gchar *sidecar_env = g_getenv ("AID_DSPOSEPATCHPERSISTENT_SIDECAR_PATH");
  if (sidecar_env && *sidecar_env)
    self->sidecar_path = g_strdup (sidecar_env);

  if (!self->track_states)
    self->track_states = new std::unordered_map<std::string, PersistentTrackState> ();
  if (!self->prev_gray)
    self->prev_gray = new cv::Mat ();

  if (cudaSetDevice (self->gpu_id) != cudaSuccess) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("Unable to set cuda device"), (NULL));
    return FALSE;
  }

  int val = -1;
  cudaDeviceGetAttribute (&val, cudaDevAttrIntegrated, self->gpu_id);
  self->is_integrated = val;

  if (cudaStreamCreate (&self->cuda_stream) != cudaSuccess) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("Could not create cuda stream"), (NULL));
    return FALSE;
  }

  NvBufSurfaceCreateParams create_params;
  std::memset (&create_params, 0, sizeof (create_params));
  create_params.gpuId = self->gpu_id;
  create_params.width = self->processing_width;
  create_params.height = self->processing_height;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = self->is_integrated ? NVBUF_MEM_DEFAULT : NVBUF_MEM_CUDA_PINNED;

  if (NvBufSurfaceCreate (&self->inter_buf, 1, &create_params) != 0) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("Could not allocate internal assist buffer"), (NULL));
    return FALSE;
  }

  self->transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
  self->transform_config_params.gpu_id = self->gpu_id;
  self->transform_config_params.cuda_stream = self->cuda_stream;

  if (self->sidecar_path) {
    self->sidecar_stream = new std::ofstream (self->sidecar_path, std::ios::out | std::ios::trunc);
    if (self->sidecar_stream->is_open ()) {
      (*self->sidecar_stream)
          << "frame_num,source_id,track_id,mode,proxy_active,proxy_age,event,stop_reason,handoff_reason,"
          << "proxy_left,proxy_top,proxy_width,proxy_height,patch_left,patch_top,patch_width,patch_height,"
          << "patch_source,tracked_points,flow_dx,flow_dy,flow_mag\n";
      self->sidecar_stream->flush ();
    } else {
      GST_WARNING_OBJECT (self, "could not open sidecar path: %s", self->sidecar_path);
      delete self->sidecar_stream;
      self->sidecar_stream = NULL;
    }
  }

  GST_INFO_OBJECT (self,
      "pose metadata is not available in the current DeepStream baseline; "
      "using bbox_fallback upper-body patch mode for persistent debug tracking");
  return TRUE;
}

static gboolean
gst_dsposepatchpersistent_stop (GstBaseTransform * btrans)
{
  GstDsPosePatchPersistent *self = GST_DSPOSEPATCHPERSISTENT (btrans);
  if (self->track_states) {
    delete self->track_states;
    self->track_states = NULL;
  }
  if (self->prev_gray) {
    delete self->prev_gray;
    self->prev_gray = NULL;
  }
  if (self->inter_buf) {
    NvBufSurfaceDestroy (self->inter_buf);
    self->inter_buf = NULL;
  }
  if (self->cuda_stream) {
    cudaStreamDestroy (self->cuda_stream);
    self->cuda_stream = NULL;
  }
  if (self->sidecar_stream) {
    self->sidecar_stream->close ();
    delete self->sidecar_stream;
    self->sidecar_stream = NULL;
  }
  if (self->sidecar_path) {
    g_free (self->sidecar_path);
    self->sidecar_path = NULL;
  }
  return TRUE;
}

static gboolean
gst_dsposepatchpersistent_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsPosePatchPersistent *self = GST_DSPOSEPATCHPERSISTENT (btrans);
  gst_video_info_from_caps (&self->video_info, incaps);
  return TRUE;
}

static GstFlowReturn
gst_dsposepatchpersistent_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstDsPosePatchPersistent *self = GST_DSPOSEPATCHPERSISTENT (btrans);

  GstMapInfo in_map_info;
  std::memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (self, STREAM, FAILED,
        ("%s: gst_buffer_map failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }

  NvBufSurface *in_surf = (NvBufSurface *) in_map_info.data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (!batch_meta) {
    gst_buffer_unmap (inbuf, &in_map_info);
    return GST_FLOW_OK;
  }

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    if (!frame_meta)
      continue;

    cv::Mat current_gray;
    if (convert_frame_to_gray (self, in_surf, frame_meta->batch_id, current_gray) != GST_FLOW_OK) {
      gst_buffer_unmap (inbuf, &in_map_info);
      return GST_FLOW_ERROR;
    }

    std::unordered_map<std::string, bool> seen_this_frame;
    std::vector<cv::Rect2f> real_boxes_proc;

    for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
      if (!obj_meta)
        continue;
      if ((gint) obj_meta->class_id != self->person_class_id)
        continue;
      if (obj_meta->object_id == UNTRACKED_OBJECT_ID)
        continue;

      const std::string key = make_track_key (frame_meta->source_id, obj_meta->object_id);
      cv::Rect2f bbox_proc = original_to_proc_rect (self, obj_meta->rect_params);
      real_boxes_proc.push_back (bbox_proc);
      seen_this_frame[key] = true;

      PersistentPatchGeometry patch_geometry;
      std::string patch_reason;
      const bool patch_ok = build_patch_geometry (self, obj_meta, bbox_proc, patch_geometry, patch_reason);

      PersistentTrackState &state = (*self->track_states)[key];
      state.source_id = frame_meta->source_id;
      state.track_id = obj_meta->object_id;
      state.last_real_frame_num = frame_meta->frame_num;
      state.proxy_age_frames = 0;
      state.last_tracked_points = 0;
      state.frozen_hold_remaining_frames = 0;
      state.proxy_active = false;
      state.frozen_hold_active = false;
      state.last_real_bbox_proc = bbox_proc;
      state.last_bbox_proc = bbox_proc;
      state.last_flow_delta_proc = cv::Point2f (0.0f, 0.0f);
      state.last_flow_mag_proc = 0.0f;
      state.last_failure_reason.clear ();
      state.patch_source = patch_ok ? patch_geometry.source : "bbox_fallback";

      if (patch_ok) {
        state.last_patch_rect_proc = patch_geometry.patch_rect_proc;
        init_features_for_patch (current_gray, patch_geometry.patch_rect_proc,
            self->feature_max_corners, state.patch_points);
        state.last_tracked_points = (guint) state.patch_points.size ();
      } else {
        state.last_patch_rect_proc = cv::Rect2f ();
        state.patch_points.clear ();
      }

      write_sidecar_row (self, frame_meta, state.track_id, "real", FALSE, 0,
          "real_update", patch_reason, "", bbox_proc,
          patch_ok ? patch_geometry.patch_rect_proc : cv::Rect2f (),
          state.patch_source, state.last_tracked_points,
          state.last_flow_delta_proc, state.last_flow_mag_proc);
    }

    const float max_shift_proc = proc_shift_limit (self);
    for (auto it = self->track_states->begin (); it != self->track_states->end ();) {
      const std::string &key = it->first;
      PersistentTrackState &state = it->second;

      if (state.source_id != frame_meta->source_id) {
        ++it;
        continue;
      }
      if (seen_this_frame.find (key) != seen_this_frame.end ()) {
        ++it;
        continue;
      }

      const guint64 miss_frames = frame_meta->frame_num > state.last_real_frame_num ?
          (frame_meta->frame_num - state.last_real_frame_num) : 0;
      if (miss_frames == 0 || miss_frames > self->hard_max_proxy_age_frames) {
        if (state.proxy_active || state.frozen_hold_active) {
          write_sidecar_row (self, frame_meta, state.track_id,
              state.frozen_hold_active ? "frozen_hold" : "proxy",
              FALSE, state.proxy_age_frames, "proxy_stop", "hard_max_age_reached", "",
              state.last_bbox_proc, state.last_patch_rect_proc, state.patch_source,
              state.last_tracked_points, state.last_flow_delta_proc, state.last_flow_mag_proc);
        }
        it = self->track_states->erase (it);
        continue;
      }

      bool handed_off = false;
      for (const cv::Rect2f &real_box : real_boxes_proc) {
        if (is_plausible_handoff (real_box, state.last_bbox_proc, max_shift_proc)) {
          write_sidecar_row (self, frame_meta, state.track_id,
              state.frozen_hold_active ? "frozen_hold" : "proxy",
              FALSE, state.proxy_age_frames, "proxy_handoff", "", "reacquired_real",
              state.last_bbox_proc, state.last_patch_rect_proc, state.patch_source,
              state.last_tracked_points, state.last_flow_delta_proc, state.last_flow_mag_proc);
          it = self->track_states->erase (it);
          handed_off = true;
          break;
        }
      }
      if (handed_off)
        continue;

      state.proxy_age_frames = (guint) miss_frames;

      if (state.frozen_hold_active) {
        if (state.frozen_hold_remaining_frames == 0) {
          write_sidecar_row (self, frame_meta, state.track_id, "frozen_hold", FALSE,
              state.proxy_age_frames, "proxy_stop", state.last_failure_reason, "",
              state.last_bbox_proc, state.last_patch_rect_proc, state.patch_source,
              state.last_tracked_points, state.last_flow_delta_proc, state.last_flow_mag_proc);
          it = self->track_states->erase (it);
          continue;
        }

        add_proxy_overlay (self, batch_meta, frame_meta, state.track_id,
            state.proxy_age_frames, "frozen_hold",
            state.last_bbox_proc, state.last_patch_rect_proc,
            state.last_tracked_points, state.patch_points);
        write_sidecar_row (self, frame_meta, state.track_id, "frozen_hold", TRUE,
            state.proxy_age_frames, "frozen_hold_continue", state.last_failure_reason, "",
            state.last_bbox_proc, state.last_patch_rect_proc, state.patch_source,
            state.last_tracked_points, state.last_flow_delta_proc, state.last_flow_mag_proc);
        state.frozen_hold_remaining_frames -= 1;
        ++it;
        continue;
      }

      if (!self->prev_gray || self->prev_gray->empty ()) {
        ++it;
        continue;
      }

      cv::Rect2f next_bbox_proc;
      cv::Rect2f next_patch_proc;
      std::vector<cv::Point2f> next_points;
      guint good_points = 0;
      cv::Point2f flow_delta_proc (0.0f, 0.0f);
      float flow_mag_proc = 0.0f;
      std::string failure_reason;
      if (!advance_patch_proxy (self, *self->prev_gray, current_gray, state,
              next_bbox_proc, next_patch_proc, next_points, good_points,
              flow_delta_proc, flow_mag_proc, failure_reason)) {
        state.patch_points.clear ();
        state.last_tracked_points = good_points;
        state.last_failure_reason = failure_reason;
        state.last_flow_delta_proc = cv::Point2f (0.0f, 0.0f);
        state.last_flow_mag_proc = 0.0f;

        if (self->freeze_on_patch_fail && self->hold_after_fail_frames > 0 &&
            (state.proxy_active || is_valid_rect (state.last_bbox_proc))) {
          state.frozen_hold_active = true;
          state.proxy_active = true;
          state.frozen_hold_remaining_frames = self->hold_after_fail_frames - 1;
          add_proxy_overlay (self, batch_meta, frame_meta, state.track_id,
              state.proxy_age_frames, "frozen_hold",
              state.last_bbox_proc, state.last_patch_rect_proc,
              state.last_tracked_points, state.patch_points);
          write_sidecar_row (self, frame_meta, state.track_id, "frozen_hold", TRUE,
              state.proxy_age_frames, "frozen_hold_start", failure_reason, "",
              state.last_bbox_proc, state.last_patch_rect_proc, state.patch_source,
              state.last_tracked_points, state.last_flow_delta_proc, state.last_flow_mag_proc);
          ++it;
          continue;
        }

        write_sidecar_row (self, frame_meta, state.track_id, "proxy", FALSE,
            state.proxy_age_frames, "proxy_stop", failure_reason, "",
            state.last_bbox_proc, state.last_patch_rect_proc, state.patch_source,
            state.last_tracked_points, state.last_flow_delta_proc, state.last_flow_mag_proc);
        it = self->track_states->erase (it);
        continue;
      }

      bool real_handoff_after_flow = false;
      for (const cv::Rect2f &real_box : real_boxes_proc) {
        if (is_plausible_handoff (real_box, next_bbox_proc, max_shift_proc)) {
          write_sidecar_row (self, frame_meta, state.track_id, "proxy", FALSE,
              state.proxy_age_frames, "proxy_handoff", "", "reacquired_real",
              next_bbox_proc, next_patch_proc, state.patch_source,
              good_points, flow_delta_proc, flow_mag_proc);
          it = self->track_states->erase (it);
          real_handoff_after_flow = true;
          break;
        }
      }
      if (real_handoff_after_flow)
        continue;

      state.proxy_active = true;
      state.frozen_hold_active = false;
      state.frozen_hold_remaining_frames = 0;
      state.last_bbox_proc = next_bbox_proc;
      state.last_patch_rect_proc = next_patch_proc;
      state.patch_points = next_points;
      state.last_tracked_points = good_points;
      state.last_flow_delta_proc = flow_delta_proc;
      state.last_flow_mag_proc = flow_mag_proc;
      state.last_failure_reason.clear ();

      add_proxy_overlay (self, batch_meta, frame_meta, state.track_id,
          state.proxy_age_frames, "proxy", state.last_bbox_proc, state.last_patch_rect_proc,
          state.last_tracked_points, state.patch_points);
      write_sidecar_row (self, frame_meta, state.track_id, "proxy", TRUE,
          state.proxy_age_frames, "proxy_active", "", "",
          state.last_bbox_proc, state.last_patch_rect_proc, state.patch_source,
          state.last_tracked_points, state.last_flow_delta_proc, state.last_flow_mag_proc);
      ++it;
    }

    if (!self->prev_gray)
      self->prev_gray = new cv::Mat ();
    current_gray.copyTo (*self->prev_gray);
  }

  gst_buffer_unmap (inbuf, &in_map_info);
  return GST_FLOW_OK;
}

static gboolean
dsposepatchpersistent_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsposepatchpersistent_debug, "dsposepatchpersistent", 0,
      "dsposepatchpersistent plugin");

  if (!gst_element_register (plugin, "dsposepatchpersistent", GST_RANK_PRIMARY,
          GST_TYPE_DSPOSEPATCHPERSISTENT))
    return FALSE;

  gst_element_register (plugin, "dsexample", GST_RANK_NONE,
      GST_TYPE_DSPOSEPATCHPERSISTENT);
  return TRUE;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsposepatchpersistent,
    DESCRIPTION, dsposepatchpersistent_plugin_init, "8.0", LICENSE, BINARY_PACKAGE, URL)
