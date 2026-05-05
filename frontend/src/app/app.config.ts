import {
  ApplicationConfig,
  ErrorHandler,
  provideBrowserGlobalErrorListeners,
} from '@angular/core';
import { provideHttpClient, withFetch } from '@angular/common/http';

class AppErrorHandler extends ErrorHandler {
  override handleError(error: unknown): void {
    console.error('[GlobalError]', error);
    super.handleError(error);
  }
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideHttpClient(withFetch()),
    { provide: ErrorHandler, useClass: AppErrorHandler },
  ],
};
