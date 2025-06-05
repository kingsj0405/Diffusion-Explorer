import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	optimizeDeps: {
		exclude: [
			'@mapbox/node-pre-gyp',
			'@mswjs/interceptors'
		]
	},
	server: {
		port: 8943,
		host: true,
		allowedHosts: [
			's33.ciplab.kr',
			'78a0-165-132-140-84.ngrok-free.app',
		]
	}
});
