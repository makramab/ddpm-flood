import { Link } from '@tanstack/react-router'
import { Waves } from 'lucide-react'

export function Hero() {
  return (
    <section className="relative py-16 md:py-24 px-4 md:px-6 text-center overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-purple-500/10" />
      <div className="relative max-w-3xl mx-auto">
        <Waves size={48} className="text-cyan-400 mx-auto mb-4" />
        <h1 className="text-4xl md:text-5xl font-black text-white mb-4">
          DDPM Flood Prediction
        </h1>
        <p className="text-lg md:text-xl text-gray-300 mb-3">
          Evaluating diffusion-based surge prediction on the Outer Banks, NC
        </p>
        <p className="text-sm text-gray-400 max-w-xl mx-auto mb-8">
          Explore how a Denoising Diffusion Probabilistic Model performs across three scenarios: in-distribution success, extrapolation failure, and cross-storm generalization failure.
        </p>
        <Link
          to="/map"
          className="inline-block px-8 py-3 bg-cyan-500 hover:bg-cyan-600 text-white font-semibold rounded-lg transition-colors shadow-lg shadow-cyan-500/30"
        >
          Explore the Map
        </Link>
      </div>
    </section>
  )
}
