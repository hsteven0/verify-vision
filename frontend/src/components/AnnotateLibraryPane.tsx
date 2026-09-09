import { useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent } from "react";

import { api } from "../api";
import { imageStateLabel, titleCase } from "../annotationUi";
import type { Project, ProjectImage, ProjectSummary, UUID } from "../types";
import { Icon } from "./Icon";
import { IMAGE_ROW_HEIGHT, imageLibraryWindow, VIRTUALIZE_IMAGE_LIBRARY_AFTER } from "./imageLibraryWindow";
import { disambiguateProjectNames } from "./projectOptionLabels";

interface ImageListProps {
  project: Project;
  projects: ProjectSummary[];
  selectedImageId: UUID | null;
  busy: boolean;
  inferenceBusy: boolean;
  onOpenProject: (projectId: UUID) => Promise<void>;
  onCreateProject: (name: string) => Promise<void>;
  onSelectImage: (imageId: UUID) => void;
  onImport: () => void;
}

/** Project and image navigation for Annotate. */
export function ImageList(props: ImageListProps) {
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const projectOptionLabels = useMemo(() => disambiguateProjectNames(props.projects), [props.projects]);

  return (
    <aside className="annotate-library-pane" aria-label="Annotation project library">
      <section className="sidebar-project-panel" aria-label="Current project">
        <label htmlFor="sidebar-project-select">Project</label>
        <div className="sidebar-project-controls">
          <select
            id="sidebar-project-select"
            value={props.project.id}
            onChange={(event) => void props.onOpenProject(event.target.value)}
            disabled={props.busy || props.inferenceBusy}
            title={props.project.name}
          >
            {props.projects.map((item) => (
              <option key={item.id} value={item.id}>
                {projectOptionLabels.get(item.id) ?? item.name}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setNewProjectOpen(true)}
            aria-label="Create another project"
            title="Create project"
            disabled={props.inferenceBusy}
          >
            <Icon name="plus" />
          </button>
        </div>
      </section>

      <ImageLibrary
        project={props.project}
        selectedImageId={props.selectedImageId}
        onSelectImage={props.onSelectImage}
        onImport={props.onImport}
      />
      {newProjectOpen && (
        <NewProjectDialog busy={props.busy} onClose={() => setNewProjectOpen(false)} onCreate={props.onCreateProject} />
      )}
    </aside>
  );
}

function NewProjectDialog({
  busy,
  onClose,
  onCreate,
}: {
  busy: boolean;
  onClose: () => void;
  onCreate: (name: string) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const dialogRef = useRef<HTMLDialogElement>(null);
  const previousFocusRef = useRef(document.activeElement instanceof HTMLElement ? document.activeElement : null);

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousFocus = previousFocusRef.current;
    dialog?.showModal();
    dialog?.querySelector<HTMLInputElement>("input")?.focus();
    return () => {
      if (dialog?.open) dialog.close();
      previousFocus?.focus();
    };
  }, []);

  const close = () => {
    dialogRef.current?.close();
    onClose();
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const projectName = name.trim();
    if (!projectName) return;
    await onCreate(projectName);
    close();
  };

  return (
    <dialog
      ref={dialogRef}
      className="modal-layer new-project-dialog"
      aria-labelledby="new-project-title"
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <section className="modal-panel new-project-panel">
        <header>
          <h2 id="new-project-title">New project</h2>
          <button type="button" onClick={close} aria-label="Close new project">
            <Icon name="close" />
          </button>
        </header>
        <form className="new-project-form" aria-label="Create project" onSubmit={(event) => void submit(event)}>
          <label htmlFor="new-project-name">Name</label>
          <input
            id="new-project-name"
            autoFocus
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. Urban cyclists"
            maxLength={120}
          />
          <footer>
            <button className="text-button" type="button" onClick={close}>
              Cancel
            </button>
            <button className="primary-button" type="submit" disabled={busy || !name.trim()}>
              {busy ? "Creating…" : "Create"}
            </button>
          </footer>
        </form>
      </section>
    </dialog>
  );
}

function ImageLibrary({
  project,
  selectedImageId,
  onSelectImage,
  onImport,
}: {
  project: Project;
  selectedImageId: UUID | null;
  onSelectImage: (imageId: UUID) => void;
  onImport: () => void;
}) {
  const [query, setQuery] = useState("");
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(500);
  const listRef = useRef<HTMLDivElement>(null);
  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    return project.images
      .map((image, index) => ({ image, index }))
      .filter(
        ({ image, index }) =>
          !normalized ||
          image.filename.toLocaleLowerCase().includes(normalized) ||
          String(index + 1).includes(normalized),
      );
  }, [project.images, query]);
  const window = imageLibraryWindow(filtered.length, scrollTop, viewportHeight);
  const visible = filtered.slice(window.start, window.end);
  const virtualized = filtered.length > VIRTUALIZE_IMAGE_LIBRARY_AFTER;

  useEffect(() => {
    setQuery("");
    setScrollTop(0);
  }, [project.id]);

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const updateHeight = () => setViewportHeight(list.clientHeight || 500);
    updateHeight();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateHeight);
    observer.observe(list);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!virtualized || !selectedImageId) return;
    const selectedIndex = filtered.findIndex(({ image }) => image.id === selectedImageId);
    const list = listRef.current;
    if (selectedIndex < 0 || !list) return;
    const top = selectedIndex * IMAGE_ROW_HEIGHT;
    if (top < list.scrollTop || top + IMAGE_ROW_HEIGHT > list.scrollTop + list.clientHeight) {
      list.scrollTop = Math.max(0, top - list.clientHeight / 2);
      setScrollTop(list.scrollTop);
    }
  }, [filtered, selectedImageId, virtualized]);

  return (
    <section className="sidebar-image-library" aria-label="Project images">
      <header>
        <div>
          <strong>Images</strong>
          <span>{project.images.length}</span>
        </div>
        <button type="button" onClick={onImport} aria-label="Import images" title="Import images">
          <Icon name="import" />
        </button>
      </header>
      {project.images.length > 9 && (
        <label className="sidebar-image-search">
          <Icon name="search" />
          <span className="visually-hidden">Search images</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search images…" />
        </label>
      )}
      <div
        ref={listRef}
        className={`sidebar-image-list ${virtualized ? "virtualized" : ""}`}
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
      >
        {filtered.length === 0 ? (
          <p>{project.images.length === 0 ? "Import images to begin." : "No images match."}</p>
        ) : virtualized ? (
          <div className="sidebar-image-virtual-space" style={{ height: `${filtered.length * IMAGE_ROW_HEIGHT}px` }}>
            {visible.map(({ image }, offset) => (
              <ImageLibraryItem
                key={image.id}
                projectId={project.id}
                image={image}
                active={image.id === selectedImageId}
                style={{ top: `${(window.start + offset) * IMAGE_ROW_HEIGHT}px` }}
                onSelect={() => onSelectImage(image.id)}
              />
            ))}
          </div>
        ) : (
          visible.map(({ image }) => (
            <ImageLibraryItem
              key={image.id}
              projectId={project.id}
              image={image}
              active={image.id === selectedImageId}
              onSelect={() => onSelectImage(image.id)}
            />
          ))
        )}
      </div>
    </section>
  );
}

function ImageLibraryItem({
  projectId,
  image,
  active,
  style,
  onSelect,
}: {
  projectId: UUID;
  image: ProjectImage;
  active: boolean;
  style?: CSSProperties;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`sidebar-image-item ${active ? "active" : ""}`}
      style={style}
      onClick={onSelect}
      aria-current={active ? "true" : undefined}
      title={image.filename}
    >
      <img src={api.imageUrl(projectId, image.id)} alt="" loading="lazy" decoding="async" />
      <span>
        <strong>{image.filename}</strong>
        <span className="sidebar-image-meta">
          <small>
            {image.annotations.length} {image.annotations.length === 1 ? "annotation" : "annotations"}
          </small>
          <span
            className={`sidebar-image-state state-${image.review_state}`}
            title={titleCase(imageStateLabel(image.review_state))}
          >
            {image.review_state === "complete" ? <Icon name="check" /> : <span className="status-dot" />}
            <span className="visually-hidden">{titleCase(imageStateLabel(image.review_state))}</span>
          </span>
        </span>
      </span>
    </button>
  );
}
