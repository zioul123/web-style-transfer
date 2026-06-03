import { useEffect, useState } from "react";
import { AdvancedOptionsPanel } from "./features/style-transfer/components/AdvancedOptionsPanel";
import { AppHeader } from "./features/style-transfer/components/AppHeader";
import { BlogTeaserPanel } from "./features/style-transfer/components/BlogTeaserPanel";
import { OptionsTogglePanel } from "./features/style-transfer/components/OptionsTogglePanel";
import { PreviewPanel } from "./features/style-transfer/components/PreviewPanel";
import { RunControlsPanel } from "./features/style-transfer/components/RunControlsPanel";
import { StatsPanel } from "./features/style-transfer/components/StatsPanel";
import { StatusPanel } from "./features/style-transfer/components/StatusPanel";
import { useStyleTransferController } from "./features/style-transfer/hooks/useStyleTransferController";
import { VGG_PACK_OPTIONS } from "./features/style-transfer/modelPacks";
import {
  isHostedVggPackName,
  isLocalhost,
} from "./features/style-transfer/uiOptions";
import { assetUrl } from "./shared/assetUrls";

function App() {
  const { controls, status, images, canRun, setIsRunning, onUpload } =
    useStyleTransferController();
  const { selectedPack, setSelectedPack } = controls;
  const [showOptions, setShowOptions] = useState<boolean>(false);
  const benchmarkUrl = assetUrl("benchmark");
  const canUseAllModelPacks = isLocalhost();
  const modelPackOptions = canUseAllModelPacks
    ? VGG_PACK_OPTIONS
    : VGG_PACK_OPTIONS.filter((option) => isHostedVggPackName(option.name));

  useEffect(() => {
    if (canUseAllModelPacks || isHostedVggPackName(selectedPack)) {
      return;
    }
    setSelectedPack("int8log-per-channel");
  }, [canUseAllModelPacks, selectedPack, setSelectedPack]);

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.18),_transparent_34rem),linear-gradient(135deg,_#020617_0%,_#111827_48%,_#0f172a_100%)] text-slate-100">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-5 sm:px-6 sm:py-8">
        <AppHeader status={status} />
        <PreviewPanel controls={controls} images={images} onUpload={onUpload} />
        <StatusPanel status={status} />
        <OptionsTogglePanel
          benchmarkUrl={benchmarkUrl}
          showOptions={showOptions}
          setShowOptions={setShowOptions}
        />
        {showOptions ? (
          <AdvancedOptionsPanel
            controls={controls}
            modelPackOptions={modelPackOptions}
            onUpload={onUpload}
            status={status}
          />
        ) : null}
        <RunControlsPanel
          canRun={canRun}
          controls={controls}
          images={images}
          setIsRunning={setIsRunning}
          status={status}
        />
        <StatsPanel controls={controls} status={status} />
        <BlogTeaserPanel />
      </div>
    </main>
  );
}

export default App;
