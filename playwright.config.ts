import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4173',
    port: 4173,
    reuseExistingServer: true
  },
  use: {
    baseURL: 'http://127.0.0.1:4173'
  },
  projects: [
    {
      name: 'chromium-swiftshader',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--use-angle=swiftshader',
            '--enable-unsafe-webgpu',
            '--enable-webgpu-developer-features'
          ]
        }
      }
    }
  ]
});
