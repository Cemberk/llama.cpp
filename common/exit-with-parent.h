// Exit when the parent process goes away (--exit-with-parent).
//
// The parent keeps a pipe open on our stdin and never writes to it. When the
// parent exits, however it exits (crash, kill -9, Task Manager), the OS closes
// its end and our read sees end of file. Then we ask for a graceful shutdown
// and, if that has not finished within a few seconds, exit anyway, so a model
// never stays loaded with nobody to use it. Works the same on Windows, Linux
// and macOS, needs no polling, and is not affected by sandboxes.
//
// Only use it when stdin is a pipe from the parent: with stdin closed or
// redirected from a file or the null device, end of file comes at once.

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>

static inline void common_exit_with_parent(std::function<void()> shutdown) {
    std::thread([shutdown = std::move(shutdown)]() {
        char buf[256];
        while (std::fread(buf, 1, sizeof(buf), stdin) > 0) {
            // the parent never writes; ignore anything that arrives
        }
        std::fprintf(stderr, "exit-with-parent: stdin closed, the parent is gone; shutting down\n");
        std::fflush(stderr);
        if (shutdown) {
            shutdown();
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::fprintf(stderr, "exit-with-parent: shutdown did not finish in 5 s; exiting\n");
        std::fflush(stderr);
        std::_Exit(1);
    }).detach();
}
