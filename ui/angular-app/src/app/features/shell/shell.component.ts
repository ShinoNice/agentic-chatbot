import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SidebarComponent } from '../sidebar/sidebar.component';
import { ChatComponent } from '../chat/chat.component';
import { InspectorComponent } from '../inspector/inspector.component';
import { ChatMessage } from '../../models/chat.model';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [CommonModule, SidebarComponent, ChatComponent, InspectorComponent],
  template: `
    <div class="flex h-screen w-full overflow-hidden text-zinc-100">
      <app-sidebar />
      <main class="flex min-w-0 flex-1">
        <div class="flex-1 min-w-0">
          <app-chat (inspect)="open($event)" />
        </div>
        @if (active()) {
          <app-inspector [msg]="active()" (close)="active.set(null)" />
        }
      </main>
    </div>
  `,
})
export class ShellComponent {
  active = signal<ChatMessage | null>(null);

  open(msg: ChatMessage): void {
    this.active.set(msg);
  }
}
