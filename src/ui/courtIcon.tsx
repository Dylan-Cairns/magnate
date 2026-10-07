import courtIcon from '../assets/icons/court.svg';
import courtFilledIcon from '../assets/icons/court-filled.svg';
import { reportImageRenderFailure } from './cardImages';

export const COURT_ICON_URL = courtIcon;
// Filled variant for dark backgrounds: black line art over a white figure fill.
export const COURT_ICON_FILLED_URL = courtFilledIcon;

export function CourtIcon({
  className,
  filled = false,
}: {
  className?: string;
  filled?: boolean;
}) {
  const src = filled ? COURT_ICON_FILLED_URL : COURT_ICON_URL;
  return (
    <img
      src={src}
      alt="Court"
      className={`court-icon${className ? ` ${className}` : ''}`}
      onError={() => reportImageRenderFailure(src, 'court rank icon')}
    />
  );
}
