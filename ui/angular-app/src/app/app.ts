import { Component } from '@angular/core';
import { ShellComponent } from './features/shell/shell.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ShellComponent],
  template: `<app-shell />`,
})
export class App {}
