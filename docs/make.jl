using Boolean
import Pkg

Pkg.add("Documenter")
using Documenter

makedocs(
	sitename = "Boolean",
	format = Documenter.HTML(),
	modules = [Boolean]
	)

	# Documenter can also automatically deploy documentation to gh-pages.
	# See "Hosting Documentation" and deploydocs() in the Documenter manual
	# for more information.
	deploydocs(
		repo = "github.com/scottrsm/Boolean.jl.git"
	)
