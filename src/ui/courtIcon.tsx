import courtIcon from '../assets/icons/court.svg';
import { reportImageRenderFailure } from './cardImages';

export const COURT_ICON_URL = courtIcon;

export function CourtIcon({ className }: { className?: string }) {
  return (
    <img
      src={COURT_ICON_URL}
      alt="Court"
      className={`court-icon${className ? ` ${className}` : ''}`}
      onError={() =>
        reportImageRenderFailure(COURT_ICON_URL, 'court rank icon')
      }
    />
  );
}
