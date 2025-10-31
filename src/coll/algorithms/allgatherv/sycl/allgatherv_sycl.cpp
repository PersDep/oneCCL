/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

namespace ccl {
namespace v1 {

ccl::event allgather_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      size_t orig_count,
                                      size_t offset,
                                      ccl::datatype dtype,
                                      ccl_comm* comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done,
                                      sycl_coll_scaleup_attr coll_attr) {
    ccl::event e;
    done = true;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const bool is_single_tile = comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected ", has_all_vertices_connected);

    uint32_t world = comm->get_node_comm()->size();
    int rank = comm->get_node_comm()->rank();

    for (uint32_t i = 0; i < recv_counts.size(); i++) {
        if (send_count != recv_counts[i]) {
            LOG_ERROR("Allgatherv only supports the case when all recv_counts are the same");
            done = false;
            return e;
        }
        assert(send_count == recv_counts[i]);
    }

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: allgatherv");
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy((char*)recv_buf + offset, send_buf, send_count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: allgatherv");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    // PCIe ring LL256
    if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()))) {
        if (!is_aligned(send_buf, recv_buf, send_count, ccl_dtype.size(), 4)) {
            done = false;
            return e;
        }
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_ll_ring", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("invoking allgatherv LL256 kernel, send_count:", send_count, " datatype: ", dtype);
        e = allgatherv_ll_ring(q,
                               send_buf,
                               send_count,
                               recv_buf,
                               recv_counts,
                               orig_count,
                               offset,
                               dtype,
                               comm,
                               global_stream,
                               deps,
                               done);
        LOG_DEBUG("invoking allgatherv LL256 kernel, send_count:",
                  send_count,
                  " datatype: ",
                  dtype,
                  done ? " done" : " not done, fallback");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        return e;
    }

    if (!ccl::global_data::env().sycl_esimd) {
        if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold) {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allgatherv_small", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
            e = allgatherv_small(q,
                                 send_buf,
                                 send_count,
                                 recv_buf,
                                 recv_counts,
                                 orig_count,
                                 offset,
                                 dtype,
                                 comm,
                                 global_stream,
                                 deps);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }
        else {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allgatherv_large", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);
            e = allgatherv_large(q,
                                 send_buf,
                                 send_count,
                                 recv_buf,
                                 recv_counts,
                                 orig_count,
                                 offset,
                                 dtype,
                                 comm,
                                 global_stream,
                                 deps,
                                 coll_attr);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }

        return e;
    }

    // ESIMD
    if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold &&
        has_all_vertices_connected) {
        init_allgatherv_small(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_small", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_small(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done)
            return e;
    }
    if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_medium_threshold &&
        !is_single_tile) {
        init_allgatherv_medium(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_medium", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_medium(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_large", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);

#if defined(CCL_SYCL_VEC_SUPPORT_FP16) && defined(CCL_SYCL_VEC_SUPPORT_BF16)
        e = allgatherv_large(
            q, send_buf, send_count, recv_buf, recv_counts, orig_count, offset, dtype, comm, global_stream, deps);
#else
        // allgatherv_large is sycl::vec based algorithm
        // when 16-bit datatypes are not supported, gather by int16 instead
        ccl::datatype new_dtype = ccl::datatype::int16;
        size_t new_send_count = send_count * ccl_dtype.size() / 2;
        ccl::vector_class<size_t> new_recv_counts;
        for (size_t i = 0; i < recv_counts.size(); i++) {
            new_recv_counts.push_back(recv_counts[i] * ccl_dtype.size() / 2);
        }
        e = allgatherv_large(send_buf,
                             new_send_count,
                             recv_buf,
                             new_recv_counts,
                             orig_count,
                             offset,
                             new_dtype,
                             comm,
                             global_stream,
                             deps);
#endif // defined(CCL_SYCL_VEC_SUPPORT_FP16) && defined(CCL_SYCL_VEC_SUPPORT_BF16)
        done = true;
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }

    return e;
}

ccl::event allgatherv_sycl_multi_node(sycl::queue& q,
                                      const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      ccl_comm* global_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    ccl::event ev;
    std::vector<event> evs;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_comm* node_comm = global_comm->get_node_comm().get();
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();
    size_t r2r_size = r2r_comm->size();
    size_t node_size = node_comm->size();

    if (r2r_size > 8) {
        // fallback to the schedule based algorithm
        LOG_DEBUG("SYCL based Allgatherv send_count = ",
                  send_count,
                  " with scaleout communicator size = ",
                  r2r_size,
                  " is not supported at the moment for the larger scale");
        done = false;
        return ev;
    }

    LOG_DEBUG("allgatherv_sycl send_count=", send_count);

    bool overlap = ccl::global_data::env().sycl_allgatherv_scaleout_overlap;
    sycl_allgatherv_tune_attr tune_attr =
        allgatherv_select_tune_attr(send_count * ccl_dtype.size(), r2r_size, ccl_dtype, use_recording_path(q));
    bool direct = tune_attr.algo == allgatherv_scaleout_algo::direct;
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev)) {
        copy_to_host = true;
    }

    ccl_buffer buf_info;
    size_t max_pack_count;
    if (direct) {
        // experimentally we found out that having a smaller host buffer that
        // is essential for direct algorithm, makes the algo run more iterations
        // that drives overlapping and we can see a performance improvements
        auto scaleout_buf_size = overlap ? std::min((size_t)134217728, global_comm->get_scaleout_host_buf_size())
                                         : global_comm->get_scaleout_host_buf_size();
        if (copy_to_host) {
            auto host_ptr = global_comm->get_scaleout_host_buf();
            buf_info.set(host_ptr, scaleout_buf_size, 0);
        }
        if (send_count * r2r_size * ccl_dtype.size() <= scaleout_buf_size) {
            max_pack_count = send_count;
        }
        else {
            max_pack_count = scaleout_buf_size / r2r_size;
            max_pack_count = max_pack_count / ccl_dtype.size();
        }
    }
    else {
        max_pack_count = send_count;
    }

    std::vector<size_t> scaleout_offsets(r2r_size);
    for (int i = 0; i < r2r_size; i++) {
        const int global_rank = r2r_comm->get_global_rank(i);
        scaleout_offsets[i] = global_rank * send_count * ccl_dtype.size();
    }

    void* scaleout_buf = recv_buf;
    sycl::event sycl_ev;
    void* scaleout_send;
    int send_offset = 0;
    int nchunks = (send_count + max_pack_count - 1) / max_pack_count;
    auto scaleup_q = sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    auto copy_q = sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    if (!overlap) {
        copy_q = scaleup_q = q;
    }
    for (int iter = 0; iter < nchunks; iter++) {
        int pack_count = (iter < nchunks - 1) ? max_pack_count : send_count - send_offset;

        // ----- Scaleout Allgatherv Phase -----
        scaleout_send = (char*)send_buf + send_offset * ccl_dtype.size();
        std::vector<size_t> recv_scaleout_counts(r2r_size, pack_count);

        ev = allgatherv_scaleout_sycl(q,
                                      scaleout_send,
                                      pack_count,
                                      scaleout_buf,
                                      recv_scaleout_counts,
                                      send_count,
                                      send_offset * ccl_dtype.size(),
                                      dtype,
                                      r2r_comm,
                                      iter == 0 ? deps : evs,
                                      iter == 0 ? true : false,
                                      done,
                                      tune_attr,
                                      copy_to_host,
                                      false,
                                      buf_info.get_ptr());
        if (!done) {
            LOG_INFO("allgatherv_sycl scaleout was not done -- falling back");
            return ev;
        }

        if (direct && copy_to_host) {
            sycl_ev = ev.get_native();
        }
        else {
            evs.clear();
            evs.push_back(std::move(ev));
        }

        // ----- Scaleup Allgatherv Inplace Phase -----
        std::vector<size_t> scaleup_counts(node_size, pack_count);
        sycl_coll_scaleup_attr coll_attr;
        coll_attr.force_use_tmp = true;
        for (int i = 0; i < r2r_size; i++) {
            if (direct && copy_to_host) {
                // even if there is just 1 rank on the node, we need to complete scaleout phase
                // by copying data into the receive buffer
                auto copy_ev = copy_q.submit([=](sycl::handler& h) {
                    h.depends_on(sycl_ev);
                    h.memcpy((char*)scaleout_buf + scaleout_offsets[i],
                             (char*)(buf_info.get_ptr()) + i * pack_count * ccl_dtype.size(),
                             pack_count * ccl_dtype.size());
                });

                evs.clear();
                evs.push_back(ccl::event::create_from_native(copy_ev));
            }

            if (node_size > 1) {
                ev = allgather_sycl_single_node(scaleup_q,
                                                (char*)scaleout_buf + scaleout_offsets[i],
                                                recv_scaleout_counts[i],
                                                (char*)recv_buf + i * node_size * send_count * ccl_dtype.size(),
                                                scaleup_counts,
                                                send_count,
                                                send_offset * ccl_dtype.size(),
                                                dtype,
                                                node_comm,
                                                global_stream,
                                                evs,
                                                done,
                                                coll_attr);

                if (!done) {
                    // fallback
                    LOG_ERROR("allgatherv_sycl allgatherv single node was not done -- falling back");
                    return ev;
                }
            }
        }

        if (iter < nchunks - 1) {
            send_offset += pack_count;
            for (int i = 0; i < r2r_size; i++) {
                scaleout_offsets[i] += pack_count * ccl_dtype.size();
            }
        }
    }

    if (node_size == 1) {
        auto sycl_evs = get_sycl_events(evs);
        auto sycl_event = submit_wait_on_events(q, sycl_evs);
        ev = ccl::event::create_from_native(sycl_event);
    }

    return ev;
}

ccl::event allgather_sycl(sycl::queue q,
                          const void* send_buf,
                          size_t send_count,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          ccl_stream* op_stream,
                          const allgatherv_attr& attr,
                          const vector_class<event>& deps,
                          bool& done) {
    auto sycl_q = op_stream->get_native_stream();

    if (send_count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(sycl_q, sycl_events);
        return ccl::event::create_from_native(e);
    }

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node) {
        LOG_DEBUG("is_single_node");
        return allgather_sycl_single_node(sycl_q,
                                          send_buf,
                                          send_count,
                                          recv_buf,
                                          recv_counts,
                                          send_count,
                                          0,
                                          dtype,
                                          comm,
                                          op_stream,
                                          deps,
                                          done);
    }

    return allgatherv_sycl_multi_node(
        sycl_q, send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, deps, done);
}

} // namespace v1
} // namespace ccl
