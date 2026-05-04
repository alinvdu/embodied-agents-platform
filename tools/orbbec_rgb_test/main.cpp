#include <libobsensor/ObSensor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    fs::path output_dir = "artifacts/orbbec_rgb";
    int frames = 30;
    int warmup_frames = 10;
    int timeout_ms = 1000;
    int width = 640;
    int height = 480;
    int fps = 30;
    int depth_width = 0;
    int depth_height = 0;
    int depth_fps = 0;
    int log_every = 1;
    int imu_log_every = 200;
    bool latest_only = false;
    bool enable_depth = false;
    bool enable_imu = false;
    bool no_file_output = false;
    bool list_profiles = false;
    bool enable_depth_registration = false;
    std::string imu_udp_host = "127.0.0.1";
    int imu_udp_port = 8766;
    bool camera_http_enable = false;
    std::string camera_http_host = "127.0.0.1";
    int camera_http_port = 8765;
    std::string camera_http_path = "/camera/rgbd";
    int camera_http_timeout_ms = 100;
    bool enable_point_cloud = false;
    std::string point_cloud_format = "xyz";
    int point_cloud_stride = 1;
    int point_cloud_max_points = 200000;
    double point_cloud_min_z_m = 0.25;
    double point_cloud_max_z_m = 4.0;
};

struct PointCloudPayload {
    uint32_t format = 0;
    uint32_t count = 0;
    uint32_t stride_bytes = 0;
    std::vector<uint8_t> bytes;
};

struct LatestImuSample {
    bool has_accel = false;
    bool has_gyro = false;
    uint64_t accel_frame_index = 0;
    uint64_t gyro_frame_index = 0;
    uint64_t accel_timestamp_us = 0;
    uint64_t gyro_timestamp_us = 0;
    float accel_temperature = 0.0f;
    float gyro_temperature = 0.0f;
    OBAccelValue accel_value{};
    OBGyroValue gyro_value{};
};

std::string imu_sample_to_json(const LatestImuSample &imu_sample) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    const uint64_t latest_imu_timestamp_us = std::max(imu_sample.accel_timestamp_us, imu_sample.gyro_timestamp_us);
    const uint64_t sample_timestamp_us =
        imu_sample.has_accel ? imu_sample.accel_timestamp_us : imu_sample.gyro_timestamp_us;
    out << "{"
        << "\"timestamp_s\":" << (static_cast<double>(sample_timestamp_us) / 1'000'000.0)
        << ",\"system_timestamp_us\":" << latest_imu_timestamp_us
        << ",\"has_accel\":" << (imu_sample.has_accel ? "true" : "false")
        << ",\"has_gyro\":" << (imu_sample.has_gyro ? "true" : "false");
    if(imu_sample.has_accel) {
        out << ",\"linear_acceleration_m_s2\":{"
            << "\"x\":" << imu_sample.accel_value.x
            << ",\"y\":" << imu_sample.accel_value.y
            << ",\"z\":" << imu_sample.accel_value.z
            << "}"
            << ",\"accel_frame_index\":" << imu_sample.accel_frame_index
            << ",\"accel_timestamp_us\":" << imu_sample.accel_timestamp_us
            << ",\"accel_temperature_c\":" << imu_sample.accel_temperature;
    }
    if(imu_sample.has_gyro) {
        out << ",\"angular_velocity_rad_s\":{"
            << "\"x\":" << imu_sample.gyro_value.x
            << ",\"y\":" << imu_sample.gyro_value.y
            << ",\"z\":" << imu_sample.gyro_value.z
            << "}"
            << ",\"gyro_frame_index\":" << imu_sample.gyro_frame_index
            << ",\"gyro_timestamp_us\":" << imu_sample.gyro_timestamp_us
            << ",\"gyro_temperature_c\":" << imu_sample.gyro_temperature;
    }
    out << "}";
    return out.str();
}

class ImuDatagramPublisher {
public:
    ImuDatagramPublisher(const std::string &host, int port) : socket_fd_(-1) {
        socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if(socket_fd_ < 0) {
            throw std::runtime_error("Failed to create IMU UDP socket");
        }
        const int flags = fcntl(socket_fd_, F_GETFL, 0);
        if(flags < 0 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
            throw std::runtime_error("Failed to mark IMU UDP socket non-blocking");
        }
        std::memset(&destination_, 0, sizeof(destination_));
        destination_.sin_family = AF_INET;
        destination_.sin_port = htons(static_cast<uint16_t>(port));
        if(::inet_pton(AF_INET, host.c_str(), &destination_.sin_addr) != 1) {
            ::close(socket_fd_);
            socket_fd_ = -1;
            throw std::runtime_error("Invalid IMU UDP host: " + host);
        }
    }

    ImuDatagramPublisher(const ImuDatagramPublisher &) = delete;
    ImuDatagramPublisher &operator=(const ImuDatagramPublisher &) = delete;

    ~ImuDatagramPublisher() {
        if(socket_fd_ >= 0) {
            ::close(socket_fd_);
        }
    }

    bool publish(const LatestImuSample &imu_sample) {
        last_error_message_.clear();
        const std::string payload = imu_sample_to_json(imu_sample);
        const ssize_t sent = ::sendto(
            socket_fd_,
            payload.data(),
            payload.size(),
            0,
            reinterpret_cast<const sockaddr *>(&destination_),
            sizeof(destination_)
        );
        if(sent >= 0) {
            return true;
        }
        if(errno == EWOULDBLOCK || errno == EAGAIN) {
            return false;
        }
        last_error_message_ = std::strerror(errno);
        return false;
    }

    const std::string &last_error_message() const {
        return last_error_message_;
    }

private:
    int socket_fd_;
    sockaddr_in destination_{};
    std::string last_error_message_;
};

void append_u32_be(std::vector<uint8_t> &buffer, uint32_t value) {
    buffer.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
    buffer.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    buffer.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    buffer.push_back(static_cast<uint8_t>(value & 0xff));
}

void append_u64_be(std::vector<uint8_t> &buffer, uint64_t value) {
    for(int shift = 56; shift >= 0; shift -= 8) {
        buffer.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
    }
}

void append_f32_le(std::vector<uint8_t> &buffer, float value) {
    uint32_t raw = 0;
    static_assert(sizeof(raw) == sizeof(value), "float32 payload requires 32-bit floats");
    std::memcpy(&raw, &value, sizeof(value));
    buffer.push_back(static_cast<uint8_t>(raw & 0xff));
    buffer.push_back(static_cast<uint8_t>((raw >> 8) & 0xff));
    buffer.push_back(static_cast<uint8_t>((raw >> 16) & 0xff));
    buffer.push_back(static_cast<uint8_t>((raw >> 24) & 0xff));
}

uint64_t unix_time_us() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::vector<uint8_t> depth_frame_to_big_endian_mm(const std::shared_ptr<ob::DepthFrame> &depth_frame) {
    const auto width = depth_frame->getWidth();
    const auto height = depth_frame->getHeight();
    const auto pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    const auto expected_size = pixel_count * sizeof(uint16_t);
    if(depth_frame->getDataSize() < expected_size) {
        throw std::runtime_error("Depth frame data is smaller than width * height * uint16");
    }
    const auto *raw = reinterpret_cast<const uint16_t *>(depth_frame->getData());
    const float scale_to_mm = depth_frame->getValueScale();
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(pixel_count * 2));
    for(uint64_t index = 0; index < pixel_count; ++index) {
        const float scaled = static_cast<float>(raw[index]) * scale_to_mm;
        const auto clamped = static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, scaled)));
        out.push_back(static_cast<uint8_t>((clamped >> 8) & 0xff));
        out.push_back(static_cast<uint8_t>(clamped & 0xff));
    }
    return out;
}

std::string camera_metadata_json(const std::shared_ptr<ob::ColorFrame> &rgb_frame, const PointCloudPayload *point_cloud) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9) << "{";
    bool wrote_field = false;
    try {
        auto profile = rgb_frame->getStreamProfile();
        if(profile && profile->is<ob::VideoStreamProfile>()) {
            auto video_profile = profile->as<ob::VideoStreamProfile>();
            const auto intrinsic = video_profile->getIntrinsic();
            out << "\"camera_intrinsics\":{"
                << "\"fx\":" << intrinsic.fx << ","
                << "\"fy\":" << intrinsic.fy << ","
                << "\"cx\":" << intrinsic.cx << ","
                << "\"cy\":" << intrinsic.cy << ","
                << "\"width\":" << intrinsic.width << ","
                << "\"height\":" << intrinsic.height
                << "}";
            wrote_field = true;
        }
    }
    catch(const std::exception &) {
    }
    if(point_cloud && point_cloud->count > 0) {
        if(wrote_field) {
            out << ",";
        }
        out << "\"point_cloud\":{"
            << "\"format\":\"xyz_float32\","
            << "\"units\":\"m\","
            << "\"count\":" << point_cloud->count << ","
            << "\"stride_bytes\":" << point_cloud->stride_bytes
            << "}";
        wrote_field = true;
    }
    out << "}";
    return out.str();
}

PointCloudPayload build_xyz_point_cloud_payload(
    const std::shared_ptr<ob::Frame> &cloud_frame,
    const Options &options
) {
    PointCloudPayload payload;
    payload.format = 1;
    payload.stride_bytes = 12;
    if(!cloud_frame || cloud_frame->dataSize() == 0) {
        return payload;
    }
    const auto *points = reinterpret_cast<const OBPoint *>(cloud_frame->data());
    const size_t raw_count = cloud_frame->dataSize() / sizeof(OBPoint);
    const size_t sample_stride = static_cast<size_t>(std::max(1, options.point_cloud_stride));
    const size_t max_points = static_cast<size_t>(std::max(0, options.point_cloud_max_points));
    payload.bytes.reserve(std::min(raw_count / sample_stride + 1, max_points) * payload.stride_bytes);
    for(size_t index = 0; index < raw_count; index += sample_stride) {
        if(max_points > 0 && payload.count >= max_points) {
            break;
        }
        const float x_m = static_cast<float>(points[index].x) / 1000.0f;
        const float y_m = static_cast<float>(points[index].y) / 1000.0f;
        const float z_m = static_cast<float>(points[index].z) / 1000.0f;
        if(!std::isfinite(x_m) || !std::isfinite(y_m) || !std::isfinite(z_m) || z_m <= 0.0f) {
            continue;
        }
        if(z_m < options.point_cloud_min_z_m || z_m > options.point_cloud_max_z_m) {
            continue;
        }
        append_f32_le(payload.bytes, x_m);
        append_f32_le(payload.bytes, y_m);
        append_f32_le(payload.bytes, z_m);
        ++payload.count;
    }
    return payload;
}

std::vector<uint8_t> build_rgbd_payload(
    const std::shared_ptr<ob::ColorFrame> &rgb_frame,
    const std::shared_ptr<ob::DepthFrame> &depth_frame,
    uint64_t fallback_frame_index,
    const PointCloudPayload *point_cloud = nullptr
) {
    const auto rgb_width = rgb_frame->getWidth();
    const auto rgb_height = rgb_frame->getHeight();
    const auto rgb_size = static_cast<uint64_t>(rgb_width) * static_cast<uint64_t>(rgb_height) * 3;
    if(rgb_frame->getFormat() != OB_FORMAT_RGB) {
        throw std::runtime_error("build_rgbd_payload received a non-RGB frame");
    }
    if(rgb_frame->getDataSize() < rgb_size) {
        throw std::runtime_error("RGB frame data is smaller than width * height * 3");
    }

    std::vector<uint8_t> depth_payload;
    uint32_t depth_width = 0;
    uint32_t depth_height = 0;
    if(depth_frame) {
        depth_width = depth_frame->getWidth();
        depth_height = depth_frame->getHeight();
        depth_payload = depth_frame_to_big_endian_mm(depth_frame);
    }

    const std::string metadata_json = camera_metadata_json(rgb_frame, point_cloud);
    const uint32_t point_format = point_cloud ? point_cloud->format : 0;
    const uint32_t point_count = point_cloud ? point_cloud->count : 0;
    const uint32_t point_stride_bytes = point_cloud ? point_cloud->stride_bytes : 0;
    const uint64_t point_payload_size = point_cloud ? static_cast<uint64_t>(point_cloud->bytes.size()) : 0;
    std::vector<uint8_t> payload;
    payload.reserve(static_cast<size_t>(88 + rgb_size + depth_payload.size() + point_payload_size + metadata_json.size()));
    const char magic[8] = {'X', 'L', 'R', 'G', 'B', 'D', '1', '\0'};
    const uint64_t frame_index = rgb_frame->getIndex() > 0 ? rgb_frame->getIndex() : fallback_frame_index;
    const uint64_t timestamp_us = rgb_frame->getSystemTimeStampUs() > 0
        ? rgb_frame->getSystemTimeStampUs()
        : unix_time_us();
    payload.insert(payload.end(), magic, magic + 8);
    append_u32_be(payload, 3);
    append_u64_be(payload, frame_index);
    append_u64_be(payload, timestamp_us);
    append_u32_be(payload, rgb_width);
    append_u32_be(payload, rgb_height);
    append_u32_be(payload, depth_width);
    append_u32_be(payload, depth_height);
    append_u64_be(payload, rgb_size);
    append_u64_be(payload, static_cast<uint64_t>(depth_payload.size()));
    append_u32_be(payload, point_format);
    append_u32_be(payload, point_count);
    append_u32_be(payload, point_stride_bytes);
    append_u64_be(payload, point_payload_size);
    append_u64_be(payload, static_cast<uint64_t>(metadata_json.size()));
    const auto *rgb_data = reinterpret_cast<const uint8_t *>(rgb_frame->getData());
    payload.insert(payload.end(), rgb_data, rgb_data + rgb_size);
    payload.insert(payload.end(), depth_payload.begin(), depth_payload.end());
    if(point_cloud) {
        payload.insert(payload.end(), point_cloud->bytes.begin(), point_cloud->bytes.end());
    }
    payload.insert(payload.end(), metadata_json.begin(), metadata_json.end());
    return payload;
}

class CameraHttpPublisher {
public:
    CameraHttpPublisher(std::string host, int port, std::string path, int timeout_ms)
        : host_(std::move(host)), port_(port), path_(std::move(path)), timeout_ms_(std::max(1, timeout_ms)) {
        if(path_.empty() || path_[0] != '/') {
            path_ = "/" + path_;
        }
    }

    bool publish(const std::vector<uint8_t> &payload) {
        last_error_message_.clear();
        int socket_fd = connect_socket();
        if(socket_fd < 0) {
            return false;
        }

        std::ostringstream request;
        request << "POST " << path_ << " HTTP/1.1\r\n"
                << "Host: " << host_ << ":" << port_ << "\r\n"
                << "Content-Type: application/octet-stream\r\n"
                << "Content-Length: " << payload.size() << "\r\n"
                << "Connection: close\r\n\r\n";
        const std::string header = request.str();
        bool ok = send_all(socket_fd, reinterpret_cast<const uint8_t *>(header.data()), header.size())
            && send_all(socket_fd, payload.data(), payload.size());
        if(ok) {
            char response[128] = {};
            const ssize_t received = ::recv(socket_fd, response, sizeof(response) - 1, 0);
            if(received <= 0 || !is_success_response(response, static_cast<size_t>(received))) {
                last_error_message_ = "camera HTTP POST did not receive a 2xx response";
                ok = false;
            }
        }
        ::close(socket_fd);
        return ok;
    }

    const std::string &last_error_message() const {
        return last_error_message_;
    }

private:
    bool is_success_response(const char *response, size_t size) const {
        const std::string response_text(response, size);
        const size_t line_end = response_text.find("\r\n");
        const std::string status_line = response_text.substr(0, line_end);
        std::istringstream stream(status_line);
        std::string version;
        int status = 0;
        stream >> version >> status;
        return version.rfind("HTTP/", 0) == 0 && status >= 200 && status < 300;
    }

    int connect_socket() {
        addrinfo hints{};
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_family = AF_UNSPEC;
        addrinfo *results = nullptr;
        const std::string port_string = std::to_string(port_);
        const int rc = ::getaddrinfo(host_.c_str(), port_string.c_str(), &hints, &results);
        if(rc != 0) {
            last_error_message_ = std::string("getaddrinfo failed: ") + gai_strerror(rc);
            return -1;
        }
        int socket_fd = -1;
        for(addrinfo *item = results; item != nullptr; item = item->ai_next) {
            socket_fd = ::socket(item->ai_family, item->ai_socktype, item->ai_protocol);
            if(socket_fd < 0) {
                continue;
            }
            if(connect_with_timeout(socket_fd, item->ai_addr, item->ai_addrlen)) {
                break;
            }
            ::close(socket_fd);
            socket_fd = -1;
        }
        ::freeaddrinfo(results);
        if(socket_fd < 0) {
            last_error_message_ = "could not connect to camera HTTP endpoint";
        }
        return socket_fd;
    }

    bool connect_with_timeout(int socket_fd, const sockaddr *addr, socklen_t addr_len) {
        const int flags = fcntl(socket_fd, F_GETFL, 0);
        if(flags < 0 || fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            last_error_message_ = "could not set camera HTTP socket non-blocking";
            return false;
        }
        int rc = ::connect(socket_fd, addr, addr_len);
        if(rc == 0) {
            fcntl(socket_fd, F_SETFL, flags);
            set_socket_timeout(socket_fd);
            return true;
        }
        if(errno != EINPROGRESS) {
            last_error_message_ = std::strerror(errno);
            return false;
        }
        fd_set write_fds;
        FD_ZERO(&write_fds);
        FD_SET(socket_fd, &write_fds);
        timeval timeout = timeout_value();
        rc = ::select(socket_fd + 1, nullptr, &write_fds, nullptr, &timeout);
        if(rc <= 0) {
            last_error_message_ = rc == 0 ? "camera HTTP connect timed out" : std::strerror(errno);
            return false;
        }
        int socket_error = 0;
        socklen_t error_len = sizeof(socket_error);
        if(::getsockopt(socket_fd, SOL_SOCKET, SO_ERROR, &socket_error, &error_len) < 0 || socket_error != 0) {
            last_error_message_ = socket_error == 0 ? std::strerror(errno) : std::strerror(socket_error);
            return false;
        }
        fcntl(socket_fd, F_SETFL, flags);
        set_socket_timeout(socket_fd);
        return true;
    }

    timeval timeout_value() const {
        timeval timeout{};
        timeout.tv_sec = timeout_ms_ / 1000;
        timeout.tv_usec = (timeout_ms_ % 1000) * 1000;
        return timeout;
    }

    void set_socket_timeout(int socket_fd) const {
        timeval timeout = timeout_value();
        ::setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    }

    bool send_all(int socket_fd, const uint8_t *data, size_t size) {
        size_t sent_total = 0;
        while(sent_total < size) {
            const ssize_t sent = ::send(socket_fd, data + sent_total, size - sent_total, 0);
            if(sent <= 0) {
                last_error_message_ = std::strerror(errno);
                return false;
            }
            sent_total += static_cast<size_t>(sent);
        }
        return true;
    }

    std::string host_;
    int port_;
    std::string path_;
    int timeout_ms_;
    std::string last_error_message_;
};

int parse_int(const std::string &value, const std::string &name) {
    try {
        size_t consumed = 0;
        int parsed = std::stoi(value, &consumed);
        if(consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    }
    catch(const std::exception &) {
        throw std::runtime_error("Invalid integer for " + name + ": " + value);
    }
}

double parse_double(const std::string &value, const std::string &name) {
    try {
        size_t consumed = 0;
        double parsed = std::stod(value, &consumed);
        if(consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    }
    catch(const std::exception &) {
        throw std::runtime_error("Invalid number for " + name + ": " + value);
    }
}

Options parse_args(int argc, char **argv) {
    Options options;
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string &name) -> std::string {
            if(i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if(arg == "--output-dir") {
            options.output_dir = require_value(arg);
        }
        else if(arg == "--frames") {
            options.frames = parse_int(require_value(arg), arg);
        }
        else if(arg == "--warmup-frames") {
            options.warmup_frames = parse_int(require_value(arg), arg);
        }
        else if(arg == "--timeout-ms") {
            options.timeout_ms = parse_int(require_value(arg), arg);
        }
        else if(arg == "--width") {
            options.width = parse_int(require_value(arg), arg);
        }
        else if(arg == "--height") {
            options.height = parse_int(require_value(arg), arg);
        }
        else if(arg == "--fps") {
            options.fps = parse_int(require_value(arg), arg);
        }
        else if(arg == "--depth-width") {
            options.depth_width = parse_int(require_value(arg), arg);
        }
        else if(arg == "--depth-height") {
            options.depth_height = parse_int(require_value(arg), arg);
        }
        else if(arg == "--depth-fps") {
            options.depth_fps = parse_int(require_value(arg), arg);
        }
        else if(arg == "--log-every") {
            options.log_every = parse_int(require_value(arg), arg);
        }
        else if(arg == "--imu-log-every") {
            options.imu_log_every = parse_int(require_value(arg), arg);
        }
        else if(arg == "--latest-only") {
            options.latest_only = true;
        }
        else if(arg == "--enable-depth") {
            options.enable_depth = true;
        }
        else if(arg == "--enable-imu") {
            options.enable_imu = true;
        }
        else if(arg == "--no-file-output") {
            options.no_file_output = true;
        }
        else if(arg == "--list-profiles") {
            options.list_profiles = true;
        }
        else if(arg == "--enable-depth-registration") {
            options.enable_depth_registration = true;
            options.enable_depth = true;
        }
        else if(arg == "--imu-udp-host") {
            options.imu_udp_host = require_value(arg);
        }
        else if(arg == "--imu-udp-port") {
            options.imu_udp_port = parse_int(require_value(arg), arg);
        }
        else if(arg == "--camera-http-enable") {
            options.camera_http_enable = true;
        }
        else if(arg == "--camera-http-host") {
            options.camera_http_host = require_value(arg);
        }
        else if(arg == "--camera-http-port") {
            options.camera_http_port = parse_int(require_value(arg), arg);
        }
        else if(arg == "--camera-http-path") {
            options.camera_http_path = require_value(arg);
        }
        else if(arg == "--camera-http-timeout-ms") {
            options.camera_http_timeout_ms = parse_int(require_value(arg), arg);
        }
        else if(arg == "--enable-point-cloud") {
            options.enable_point_cloud = true;
            options.enable_depth = true;
        }
        else if(arg == "--point-cloud-format") {
            options.point_cloud_format = require_value(arg);
        }
        else if(arg == "--point-cloud-stride") {
            options.point_cloud_stride = parse_int(require_value(arg), arg);
        }
        else if(arg == "--point-cloud-max-points") {
            options.point_cloud_max_points = parse_int(require_value(arg), arg);
        }
        else if(arg == "--point-cloud-min-z-m") {
            options.point_cloud_min_z_m = parse_double(require_value(arg), arg);
        }
        else if(arg == "--point-cloud-max-z-m") {
            options.point_cloud_max_z_m = parse_double(require_value(arg), arg);
        }
        else if(arg == "--help" || arg == "-h") {
            std::cout << "Usage: orbbec_rgb_test [--output-dir DIR] [--frames N]\n"
                      << "                       [--warmup-frames N] [--timeout-ms MS]\n"
                      << "                       [--width PX] [--height PX] [--fps FPS]\n"
                      << "                       [--depth-width PX] [--depth-height PX] [--depth-fps FPS]\n"
                      << "                       [--log-every N] [--imu-log-every N]\n"
                      << "                       [--imu-udp-host HOST] [--imu-udp-port PORT]\n"
                      << "                       [--camera-http-enable] [--camera-http-host HOST]\n"
                      << "                       [--camera-http-port PORT] [--camera-http-path PATH]\n"
                      << "                       [--camera-http-timeout-ms MS]\n"
                      << "                       [--enable-point-cloud] [--point-cloud-format xyz]\n"
                      << "                       [--point-cloud-stride N] [--point-cloud-max-points N]\n"
                      << "                       [--point-cloud-min-z-m M] [--point-cloud-max-z-m M]\n"
                      << "                       [--latest-only] [--no-file-output] [--enable-depth] [--enable-imu]\n"
                      << "                       [--enable-depth-registration] [--list-profiles]\n";
            std::exit(EXIT_SUCCESS);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if(options.frames < 0) {
        throw std::runtime_error("--frames must be 0 for continuous capture, or a positive frame count");
    }
    if(options.width < 1 || options.height < 1 || options.fps < 1) {
        throw std::runtime_error("--width, --height, and --fps must be positive");
    }
    if(options.depth_width < 0 || options.depth_height < 0 || options.depth_fps < 0) {
        throw std::runtime_error("--depth-width, --depth-height, and --depth-fps must be zero/auto or positive");
    }
    if(options.log_every < 0) {
        throw std::runtime_error("--log-every must be 0 or positive");
    }
    if(options.imu_log_every < 0) {
        throw std::runtime_error("--imu-log-every must be 0 or positive");
    }
    if(options.imu_udp_port < 1 || options.imu_udp_port > 65535) {
        throw std::runtime_error("--imu-udp-port must be between 1 and 65535");
    }
    if(options.camera_http_port < 1 || options.camera_http_port > 65535) {
        throw std::runtime_error("--camera-http-port must be between 1 and 65535");
    }
    if(options.camera_http_timeout_ms < 1) {
        throw std::runtime_error("--camera-http-timeout-ms must be positive");
    }
    if(options.point_cloud_format != "xyz") {
        throw std::runtime_error("Only --point-cloud-format xyz is implemented in this milestone");
    }
    if(options.point_cloud_stride < 1) {
        throw std::runtime_error("--point-cloud-stride must be positive");
    }
    if(options.point_cloud_max_points < 0) {
        throw std::runtime_error("--point-cloud-max-points must be zero/unlimited or positive");
    }
    if(options.point_cloud_min_z_m < 0.0 || options.point_cloud_max_z_m <= options.point_cloud_min_z_m) {
        throw std::runtime_error("--point-cloud-min-z-m and --point-cloud-max-z-m must define a positive range");
    }
    if(options.enable_depth_registration && !options.enable_depth) {
        throw std::runtime_error("--enable-depth-registration requires --enable-depth");
    }
    return options;
}

std::string format_name(OBFormat format) {
    return ob::TypeHelper::convertOBFormatTypeToString(format);
}

std::string stream_name(OBStreamType type) {
    return ob::TypeHelper::convertOBStreamTypeToString(type);
}

std::string sensor_name(OBSensorType type) {
    return ob::TypeHelper::convertOBSensorTypeToString(type);
}

void print_device_profiles(const std::shared_ptr<ob::Device> &device) {
    auto sensors = device->getSensorList();
    const auto sensor_count = sensors->getCount();
    std::cout << "Available stream profiles:\n";
    for(uint32_t sensor_index = 0; sensor_index < sensor_count; ++sensor_index) {
        const auto sensor_type = sensors->getSensorType(sensor_index);
        std::cout << "  Sensor " << sensor_index << ": " << sensor_name(sensor_type) << "\n";
        auto sensor = sensors->getSensor(sensor_index);
        auto profiles = sensor->getStreamProfileList();
        const auto profile_count = profiles->getCount();
        if(profile_count == 0) {
            std::cout << "    (no profiles)\n";
            continue;
        }
        for(uint32_t profile_index = 0; profile_index < profile_count; ++profile_index) {
            auto profile = profiles->getProfile(profile_index);
            std::cout << "    [" << profile_index << "] " << stream_name(profile->getType())
                      << " format=" << format_name(profile->getFormat());
            if(profile->is<ob::VideoStreamProfile>()) {
                auto video = profile->as<ob::VideoStreamProfile>();
                std::cout << " " << video->getWidth() << "x" << video->getHeight()
                          << "@" << video->getFps();
            }
            else if(profile->is<ob::AccelStreamProfile>()) {
                auto accel = profile->as<ob::AccelStreamProfile>();
                std::cout << " sample_rate="
                          << ob::TypeHelper::convertOBIMUSampleRateTypeToString(accel->getSampleRate())
                          << " full_scale="
                          << ob::TypeHelper::convertOBAccelFullScaleRangeTypeToString(accel->getFullScaleRange());
            }
            else if(profile->is<ob::GyroStreamProfile>()) {
                auto gyro = profile->as<ob::GyroStreamProfile>();
                std::cout << " sample_rate="
                          << ob::TypeHelper::convertOBIMUSampleRateTypeToString(gyro->getSampleRate())
                          << " full_scale="
                          << ob::TypeHelper::convertOBGyroFullScaleRangeTypeToString(gyro->getFullScaleRange());
            }
            std::cout << "\n";
        }
    }
}

std::shared_ptr<ob::Config> build_pipeline_config(ob::Pipeline &pipeline, const Options &options) {
    auto config = std::make_shared<ob::Config>();
    if(options.enable_depth_registration) {
        auto color_profiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
        auto color_profile = color_profiles->getVideoStreamProfile(
            options.width,
            options.height,
            OB_FORMAT_RGB,
            options.fps
        );
        auto depth_profiles = pipeline.getD2CDepthProfileList(color_profile, ALIGN_D2C_HW_MODE);
        if(depth_profiles->getCount() == 0) {
            throw std::runtime_error("Hardware depth-to-color alignment is not supported for the selected color profile.");
        }
        const int depth_width = options.depth_width > 0 ? options.depth_width : OB_WIDTH_ANY;
        const int depth_height = options.depth_height > 0 ? options.depth_height : OB_HEIGHT_ANY;
        const int depth_fps = options.depth_fps > 0 ? options.depth_fps : options.fps;
        auto depth_profile = depth_profiles->getVideoStreamProfile(
            depth_width,
            depth_height,
            OB_FORMAT_Y16,
            depth_fps
        );
        config->enableStream(color_profile);
        config->enableStream(depth_profile);
        config->setAlignMode(ALIGN_D2C_HW_MODE);
        config->setDepthScaleRequire(true);
        config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
        pipeline.enableFrameSync();

        std::cout << "Hardware depth-to-color registration enabled: "
                  << "color=" << color_profile->getWidth() << "x" << color_profile->getHeight()
                  << "@" << color_profile->getFps()
                  << " depth_source=" << depth_profile->getWidth() << "x" << depth_profile->getHeight()
                  << "@" << depth_profile->getFps()
                  << " format=" << format_name(depth_profile->getFormat()) << "\n";
        return config;
    }

    config->enableVideoStream(
        OB_STREAM_COLOR,
        static_cast<uint32_t>(options.width),
        static_cast<uint32_t>(options.height),
        static_cast<uint32_t>(options.fps),
        OB_FORMAT_ANY
    );
    if(options.enable_depth) {
        const uint32_t depth_width = options.depth_width > 0 ? static_cast<uint32_t>(options.depth_width) : OB_WIDTH_ANY;
        const uint32_t depth_height = options.depth_height > 0 ? static_cast<uint32_t>(options.depth_height) : OB_HEIGHT_ANY;
        const uint32_t depth_fps = options.depth_fps > 0
            ? static_cast<uint32_t>(options.depth_fps)
            : static_cast<uint32_t>(options.fps);
        config->enableVideoStream(
            OB_STREAM_DEPTH,
            depth_width,
            depth_height,
            depth_fps,
            OB_FORMAT_Y16
        );
    }
    return config;
}

std::shared_ptr<ob::ColorFrame> to_rgb_frame(
    const std::shared_ptr<ob::ColorFrame> &color_frame,
    const std::shared_ptr<ob::FormatConvertFilter> &converter
) {
    if(color_frame->getFormat() == OB_FORMAT_RGB) {
        return color_frame;
    }
    if(color_frame->getFormat() == OB_FORMAT_MJPG) {
        converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
    }
    else if(color_frame->getFormat() == OB_FORMAT_UYVY) {
        converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
    }
    else if(color_frame->getFormat() == OB_FORMAT_YUYV) {
        converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
    }
    else {
        throw std::runtime_error(std::string("Unsupported color format: ") + format_name(color_frame->getFormat()));
    }

    return converter->process(color_frame)->as<ob::ColorFrame>();
}

void write_ppm(const fs::path &path, const std::shared_ptr<ob::ColorFrame> &rgb_frame) {
    const auto width = rgb_frame->getWidth();
    const auto height = rgb_frame->getHeight();
    const auto expected_size = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 3;
    if(rgb_frame->getFormat() != OB_FORMAT_RGB) {
        throw std::runtime_error("write_ppm received a non-RGB frame");
    }
    if(rgb_frame->getDataSize() < expected_size) {
        throw std::runtime_error("RGB frame data is smaller than width * height * 3");
    }

    std::ofstream out(path, std::ios::binary);
    if(!out) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(rgb_frame->getData()), static_cast<std::streamsize>(expected_size));
}

void write_ppm_atomic(const fs::path &path, const std::shared_ptr<ob::ColorFrame> &rgb_frame) {
    const auto tmp_path = path.string() + ".tmp";
    write_ppm(tmp_path, rgb_frame);
    fs::rename(tmp_path, path);
}

void write_depth_pgm_mm(const fs::path &path, const std::shared_ptr<ob::DepthFrame> &depth_frame) {
    const auto width = depth_frame->getWidth();
    const auto height = depth_frame->getHeight();
    const auto pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    const auto expected_size = pixel_count * sizeof(uint16_t);
    if(depth_frame->getDataSize() < expected_size) {
        throw std::runtime_error("Depth frame data is smaller than width * height * uint16");
    }

    const auto *raw = reinterpret_cast<const uint16_t *>(depth_frame->getData());
    const float scale_to_mm = depth_frame->getValueScale();
    std::ofstream out(path, std::ios::binary);
    if(!out) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    out << "P5\n" << width << " " << height << "\n65535\n";
    for(uint64_t index = 0; index < pixel_count; ++index) {
        const float scaled = static_cast<float>(raw[index]) * scale_to_mm;
        const auto clamped = static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, scaled)));
        const char bytes[2] = {
            static_cast<char>((clamped >> 8) & 0xff),
            static_cast<char>(clamped & 0xff),
        };
        out.write(bytes, 2);
    }
}

void write_depth_pgm_atomic(const fs::path &path, const std::shared_ptr<ob::DepthFrame> &depth_frame) {
    const auto tmp_path = path.string() + ".tmp";
    write_depth_pgm_mm(tmp_path, depth_frame);
    fs::rename(tmp_path, path);
}

void write_latest_metadata(
    const fs::path &path,
    const std::shared_ptr<ob::ColorFrame> &frame,
    const std::shared_ptr<ob::DepthFrame> &depth_frame,
    const LatestImuSample *imu_sample,
    int captured_frames,
    const std::string &source_format
) {
    const auto tmp_path = path.string() + ".tmp";
    std::ofstream out(tmp_path);
    if(!out) {
        throw std::runtime_error("Could not open metadata file: " + tmp_path);
    }
    out << "{\n"
        << "  \"captured_frames\": " << captured_frames << ",\n"
        << "  \"width\": " << frame->getWidth() << ",\n"
        << "  \"height\": " << frame->getHeight() << ",\n"
        << "  \"format\": \"RGB\",\n"
        << "  \"source_format\": \"" << source_format << "\",\n"
        << "  \"frame_index\": " << frame->getIndex() << ",\n"
        << "  \"device_timestamp_us\": " << frame->getTimeStampUs() << ",\n"
        << "  \"system_timestamp_us\": " << frame->getSystemTimeStampUs() << ",\n"
        << "  \"latest_frame\": \"latest.ppm\"";
    if(depth_frame) {
        out << ",\n"
            << "  \"depth_width\": " << depth_frame->getWidth() << ",\n"
            << "  \"depth_height\": " << depth_frame->getHeight() << ",\n"
            << "  \"depth_format\": \"PGM_U16_MM\",\n"
            << "  \"depth_value_scale\": " << depth_frame->getValueScale() << ",\n"
            << "  \"latest_depth_frame\": \"latest_depth.pgm\"\n";
    }
    else {
        out << "\n";
    }
    if(imu_sample && (imu_sample->has_accel || imu_sample->has_gyro)) {
        out << ",\n"
            << "  \"imu\": {\n"
            << "    \"has_accel\": " << (imu_sample->has_accel ? "true" : "false") << ",\n"
            << "    \"has_gyro\": " << (imu_sample->has_gyro ? "true" : "false") << ",\n";
        if(imu_sample->has_accel) {
            out << "    \"linear_acceleration_m_s2\": {\n"
                << "      \"x\": " << imu_sample->accel_value.x << ",\n"
                << "      \"y\": " << imu_sample->accel_value.y << ",\n"
                << "      \"z\": " << imu_sample->accel_value.z << "\n"
                << "    },\n"
                << "    \"accel_frame_index\": " << imu_sample->accel_frame_index << ",\n"
                << "    \"accel_timestamp_us\": " << imu_sample->accel_timestamp_us << ",\n"
                << "    \"accel_temperature_c\": " << imu_sample->accel_temperature << ",\n";
        }
        if(imu_sample->has_gyro) {
            out << "    \"angular_velocity_rad_s\": {\n"
                << "      \"x\": " << imu_sample->gyro_value.x << ",\n"
                << "      \"y\": " << imu_sample->gyro_value.y << ",\n"
                << "      \"z\": " << imu_sample->gyro_value.z << "\n"
                << "    },\n"
                << "    \"gyro_frame_index\": " << imu_sample->gyro_frame_index << ",\n"
                << "    \"gyro_timestamp_us\": " << imu_sample->gyro_timestamp_us << ",\n"
                << "    \"gyro_temperature_c\": " << imu_sample->gyro_temperature << ",\n";
        }
        const uint64_t latest_imu_timestamp_us = std::max(imu_sample->accel_timestamp_us, imu_sample->gyro_timestamp_us);
        out << "    \"system_timestamp_us\": " << latest_imu_timestamp_us << "\n"
            << "  }\n";
    }
    out
        << "}\n";
    out.close();
    fs::rename(tmp_path, path);
}

int main(int argc, char **argv) try {
    const Options options = parse_args(argc, argv);
    if(!options.no_file_output) {
        fs::create_directories(options.output_dir);
    }

    ob::Pipeline pipeline;

    auto device = pipeline.getDevice();
    auto info = device->getDeviceInfo();
    std::cout << "Orbbec device: " << info->getName()
              << " pid=0x" << std::hex << info->getPid() << std::dec
              << " sn=" << info->getSerialNumber() << "\n";
    if(options.list_profiles) {
        print_device_profiles(device);
        return EXIT_SUCCESS;
    }
    auto config = build_pipeline_config(pipeline, options);
    if(options.enable_depth) {
        std::cout << "Depth stream requested: "
                  << (options.depth_width > 0 ? std::to_string(options.depth_width) : "any")
                  << "x"
                  << (options.depth_height > 0 ? std::to_string(options.depth_height) : "any")
                  << "@"
                  << (options.depth_fps > 0 ? std::to_string(options.depth_fps) : std::to_string(options.fps))
                  << " Y16\n";
    }

    auto converter = std::make_shared<ob::FormatConvertFilter>();
    std::shared_ptr<ob::PointCloudFilter> point_cloud_filter = nullptr;
    if(options.enable_point_cloud) {
        point_cloud_filter = std::make_shared<ob::PointCloudFilter>();
        point_cloud_filter->setCreatePointFormat(OB_FORMAT_POINT);
        std::cout << "Point cloud enabled: format=xyz_float32_m"
                  << " stride=" << options.point_cloud_stride
                  << " max_points=" << options.point_cloud_max_points
                  << " z_range_m=[" << options.point_cloud_min_z_m
                  << "," << options.point_cloud_max_z_m << "]\n";
    }

    std::shared_ptr<ob::Pipeline> imu_pipeline = nullptr;
    std::unique_ptr<ImuDatagramPublisher> imu_publisher;
    std::unique_ptr<CameraHttpPublisher> camera_publisher;
    std::mutex imu_mutex;
    LatestImuSample latest_imu_sample;
    bool imu_running = false;
    uint64_t imu_callback_count = 0;
    uint64_t imu_udp_drop_count = 0;
    uint64_t imu_udp_error_count = 0;
    auto imu_log_started_at = std::chrono::steady_clock::now();
    if(options.enable_imu) {
        try {
            imu_publisher = std::make_unique<ImuDatagramPublisher>(options.imu_udp_host, options.imu_udp_port);
            std::cout << "IMU UDP publisher: " << options.imu_udp_host << ":" << options.imu_udp_port << "\n";
            if(options.frames != 0) {
                std::cerr << "WARNING: --enable-imu with finite --frames=" << options.frames
                          << " will stop the IMU UDP stream when RGB capture exits. "
                          << "Use --frames 0 for continuous robot runs.\n";
            }
            imu_pipeline = std::make_shared<ob::Pipeline>(device);
            auto imu_config = std::make_shared<ob::Config>();
            imu_config->enableGyroStream();
            imu_config->enableAccelStream();
            imu_config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
            imu_pipeline->start(imu_config, [&](std::shared_ptr<ob::FrameSet> frame_set) {
                auto accel_frame_raw = frame_set->getFrame(OB_FRAME_ACCEL);
                auto gyro_frame_raw = frame_set->getFrame(OB_FRAME_GYRO);
                LatestImuSample sample_to_publish;
                bool has_sample = false;
                {
                    std::lock_guard<std::mutex> lock(imu_mutex);
                    if(accel_frame_raw) {
                        auto accel_frame = accel_frame_raw->as<ob::AccelFrame>();
                        latest_imu_sample.has_accel = true;
                        latest_imu_sample.accel_frame_index = accel_frame->getIndex();
                        latest_imu_sample.accel_timestamp_us = accel_frame->getTimeStampUs();
                        latest_imu_sample.accel_temperature = accel_frame->getTemperature();
                        latest_imu_sample.accel_value = accel_frame->getValue();
                    }
                    if(gyro_frame_raw) {
                        auto gyro_frame = gyro_frame_raw->as<ob::GyroFrame>();
                        latest_imu_sample.has_gyro = true;
                        latest_imu_sample.gyro_frame_index = gyro_frame->getIndex();
                        latest_imu_sample.gyro_timestamp_us = gyro_frame->getTimeStampUs();
                        latest_imu_sample.gyro_temperature = gyro_frame->getTemperature();
                        latest_imu_sample.gyro_value = gyro_frame->getValue();
                    }
                    if(latest_imu_sample.has_accel || latest_imu_sample.has_gyro) {
                        sample_to_publish = latest_imu_sample;
                        has_sample = true;
                    }
                }
                if(!has_sample) {
                    return;
                }
                ++imu_callback_count;
                if(imu_publisher && !imu_publisher->publish(sample_to_publish)) {
                    if(imu_publisher->last_error_message().empty()) {
                        ++imu_udp_drop_count;
                    }
                    else {
                        ++imu_udp_error_count;
                    }
                }
                if(options.imu_log_every > 0 && imu_callback_count % static_cast<uint64_t>(options.imu_log_every) == 0) {
                    const double elapsed_s =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - imu_log_started_at).count();
                    const double rate_hz = elapsed_s > 0.0 ? static_cast<double>(imu_callback_count) / elapsed_s : 0.0;
                    std::cout << "IMU callback rate ~= " << rate_hz << " Hz"
                              << " callbacks=" << imu_callback_count
                              << " udp_drops=" << imu_udp_drop_count
                              << " udp_errors=" << imu_udp_error_count
                              << "\n";
                    imu_log_started_at = std::chrono::steady_clock::now();
                    imu_callback_count = 0;
                    imu_udp_drop_count = 0;
                    imu_udp_error_count = 0;
                    if(imu_publisher && !imu_publisher->last_error_message().empty()) {
                        std::cerr << "Latest IMU UDP error: " << imu_publisher->last_error_message() << "\n";
                    }
                }
            });
            imu_running = true;
        }
        catch(const std::exception &e) {
            std::cerr << "IMU stream unavailable: " << e.what() << "\n";
        }
    }
    if(options.camera_http_enable) {
        camera_publisher = std::make_unique<CameraHttpPublisher>(
            options.camera_http_host,
            options.camera_http_port,
            options.camera_http_path,
            options.camera_http_timeout_ms
        );
        std::cout << "Camera RGB-D HTTP publisher enabled: "
                  << options.camera_http_host << ":" << options.camera_http_port
                  << options.camera_http_path << "\n";
    }
    pipeline.start(config);

    for(int i = 0; i < options.warmup_frames; ++i) {
        pipeline.waitForFrameset(options.timeout_ms);
    }

    int captured = 0;
    const auto started_at = std::chrono::steady_clock::now();
    while(options.frames == 0 || captured < options.frames) {
        auto frame_set = pipeline.waitForFrameset(options.timeout_ms);
        if(!frame_set) {
            std::cout << "No color frames received within " << options.timeout_ms << "ms\n";
            continue;
        }

        auto frame = frame_set->getFrame(OB_FRAME_COLOR);
        if(!frame) {
            continue;
        }
        std::shared_ptr<ob::DepthFrame> depth_frame = nullptr;
        if(options.enable_depth) {
            auto depth = frame_set->getFrame(OB_FRAME_DEPTH);
            if(!depth) {
                continue;
            }
            depth_frame = depth->as<ob::DepthFrame>();
        }

        auto color_frame = frame->as<ob::ColorFrame>();
        const std::string source_format = format_name(color_frame->getFormat());
        auto rgb_frame = to_rgb_frame(color_frame, converter);
        PointCloudPayload point_cloud_payload;
        PointCloudPayload *point_cloud_ptr = nullptr;
        if(point_cloud_filter) {
            auto cloud_frame = point_cloud_filter->process(frame_set);
            point_cloud_payload = build_xyz_point_cloud_payload(cloud_frame, options);
            point_cloud_ptr = &point_cloud_payload;
        }

        ++captured;
        const auto latest_path = options.output_dir / "latest.ppm";
        if(!options.no_file_output) {
            if(!options.latest_only) {
                const auto numbered_path = options.output_dir / ("frame_" + std::to_string(captured) + ".ppm");
                write_ppm_atomic(numbered_path, rgb_frame);
            }
            write_ppm_atomic(latest_path, rgb_frame);
            if(depth_frame) {
                const auto latest_depth_path = options.output_dir / "latest_depth.pgm";
                if(!options.latest_only) {
                    const auto numbered_depth_path =
                        options.output_dir / ("frame_" + std::to_string(captured) + "_depth.pgm");
                    write_depth_pgm_atomic(numbered_depth_path, depth_frame);
                }
                write_depth_pgm_atomic(latest_depth_path, depth_frame);
            }
        }
        if(camera_publisher) {
            const auto payload = build_rgbd_payload(
                rgb_frame,
                depth_frame,
                static_cast<uint64_t>(captured),
                point_cloud_ptr
            );
            if(!camera_publisher->publish(payload) && options.log_every > 0) {
                std::cout << "WARNING: camera RGB-D HTTP publish failed: "
                          << camera_publisher->last_error_message() << "\n";
            }
        }
        LatestImuSample imu_snapshot;
        const LatestImuSample *imu_ptr = nullptr;
        if(imu_running) {
            std::lock_guard<std::mutex> lock(imu_mutex);
            imu_snapshot = latest_imu_sample;
            if(imu_snapshot.has_accel || imu_snapshot.has_gyro) {
                imu_ptr = &imu_snapshot;
            }
        }
        if(!options.no_file_output) {
            write_latest_metadata(options.output_dir / "latest.json", rgb_frame, depth_frame, imu_ptr, captured, source_format);
        }

        if(options.log_every > 0 && captured % options.log_every == 0) {
            std::cout << "Captured RGB frame " << captured
                      << " " << rgb_frame->getWidth() << "x" << rgb_frame->getHeight()
                      << " source=" << source_format
                      << (options.no_file_output ? "" : (" saved=" + latest_path.string()));
            if(depth_frame) {
                std::cout << " depth=" << depth_frame->getWidth() << "x" << depth_frame->getHeight();
                if(!options.no_file_output) {
                    std::cout << " saved=" << (options.output_dir / "latest_depth.pgm");
                }
            }
            if(point_cloud_ptr) {
                std::cout << " points=" << point_cloud_ptr->count;
            }
            std::cout << "\n";
        }
    }

    pipeline.stop();
    if(imu_running && imu_pipeline) {
        imu_pipeline->stop();
    }

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    std::cout << "Captured " << captured << " frames in " << elapsed << "s\n";
    return EXIT_SUCCESS;
}
catch(const ob::Error &e) {
    std::cerr << "Orbbec SDK error\n"
              << "function: " << e.getFunction() << "\n"
              << "args: " << e.getArgs() << "\n"
              << "message: " << e.what() << "\n"
              << "type: " << e.getExceptionType() << "\n";
    return EXIT_FAILURE;
}
catch(const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
}
