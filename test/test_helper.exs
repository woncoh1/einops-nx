ExUnit.start()

# Keep the default suite independent of optional native compilers/backends.
Nx.global_default_backend(Nx.BinaryBackend)
Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
